use super::*;
use crate::kernels::test_support::{bf16_bytes, upload, values, zeros_as};
use std::sync::Arc;

fn compare(a: &W8A8Activation, b: &W8A8Activation) {
    assert_eq!((a.rows, a.input_dim), (b.rows, b.input_dim));
    for (a, b, size) in [
        (&a.quantized, &b.quantized, a.rows * a.input_dim),
        (&a.row_scales, &b.row_scales, (a.rows + SCALE_PAD) * 2),
    ] {
        let mut x = vec![0; size];
        let mut y = vec![0; size];
        a.copy_to_host(&mut x).unwrap();
        b.copy_to_host(&mut y).unwrap();
        assert_eq!(x, y);
    }
}

fn factors(ctx: &MusaContext, cols: usize) -> Tensor {
    let bytes: Vec<u8> = (0..cols)
        .flat_map(|i| (0.25 + (i % 13) as f32 * 0.125).to_le_bytes())
        .collect();
    let storage = MusaBuffer::alloc_on(ctx, bytes.len()).unwrap();
    storage.copy_from_host(&bytes).unwrap();
    storage
        .as_tensor(Shape::new(vec![cols]), DType::F32)
        .unwrap()
}

fn raw(t: &Tensor) -> Vec<u8> {
    let mut bytes = vec![0; t.size_in_bytes()];
    MusaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
}

#[test]
#[ignore = "needs a MUSA device"]
fn encode_norm_quant_matches_split_and_graph_replay() {
    let ctx = Arc::new(MusaContext::new(0).unwrap());
    for (rows, cols, layer) in [
        (1, 2048, false),
        (522, 2048, false),
        (1, 1152, true),
        (5, 1152, true),
        (512, 1152, true),
        (3, 150, false),
        (3, 150, true),
    ] {
        let input = upload(&ctx, &values(rows * cols, 52), Shape::new(vec![rows, cols]));
        let residual = upload(&ctx, &values(rows * cols, 25), Shape::new(vec![rows, cols]));
        let weight = upload(&ctx, &values(cols, 26), Shape::new(vec![cols]));
        let bias = upload(&ctx, &values(cols, 27), Shape::new(vec![cols]));
        let nb = upload(&ctx, &values(cols, 28), Shape::new(vec![cols]));
        let factors = factors(&ctx, cols);
        for inv in [None, Some(&factors)] {
            for mode in 0..3 {
                let pb = (mode == 2).then_some(&bias);
                let r = (mode != 0).then_some((&residual, pb));
                let nb = layer.then_some(&nb);
                let aw = crate::workspace::GraphWorkspace::new(32 * 1024 * 1024, 0).unwrap();
                let bw = crate::workspace::GraphWorkspace::new(32 * 1024 * 1024, 0).unwrap();
                let (ag, (expected_hidden, expected)) = crate::graph::capture(ctx.clone(), || {
                    crate::workspace::with_workspace(&aw, || {
                        let (h, n) = match (r, nb) {
                            (Some((r, b)), Some(nb)) => {
                                let o = crate::kernels::fused::bias_residual_layer_bf16(
                                    &ctx, &input, b, r, &weight, nb, 1e-6,
                                )?;
                                (Some(o.hidden), o.normalized)
                            }
                            (Some((r, b)), None) => {
                                let o = crate::kernels::fused::bias_residual_rms_bf16(
                                    &ctx, &input, b, r, &weight, 1e-6,
                                )?;
                                (Some(o.hidden), o.normalized)
                            }
                            (None, Some(nb)) => (
                                None,
                                crate::kernels::norm::layer_bf16(&ctx, &input, &weight, nb, 1e-6)?,
                            ),
                            (None, None) => (
                                None,
                                crate::kernels::norm::rms_bf16(&ctx, &input, &weight, 1e-6)?,
                            ),
                        };
                        Ok((h, quantize_w8a8_activation(&ctx, &n, inv)?))
                    })
                })
                .unwrap();
                let (bg, (hidden, actual)) = crate::graph::capture(ctx.clone(), || {
                    crate::workspace::with_workspace(&bw, || {
                        norm_quantize_w8a8_activation(&ctx, &input, r, &weight, nb, 1e-6, inv)
                    })
                })
                .unwrap();
                for iteration in 0..6 {
                    let mut data = values(rows * cols, iteration + 40);
                    if iteration == 0 {
                        data.fill(0.0);
                    }
                    if iteration == 1 {
                        data[0] = 128.;
                        data[1] = -128.;
                    }
                    MusaBuffer::from_tensor(&input)
                        .unwrap()
                        .copy_from_host(&bf16_bytes(&data))
                        .unwrap();
                    ctx.synchronize().unwrap();
                    ag.replay().unwrap();
                    bg.replay().unwrap();
                    ctx.synchronize().unwrap();
                    compare(&actual, &expected);
                    if let (Some(a), Some(b)) = (&hidden, &expected_hidden) {
                        assert_eq!(raw(a), raw(b));
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a MUSA device"]
fn encode_norm_quant_validates_and_falls_back_for_unaligned_views() {
    let ctx = MusaContext::new(0).unwrap();
    for cols in [2048, 1152] {
        let input = upload(&ctx, &values(3 * cols, 10), Shape::new(vec![3, cols]));
        let vector = upload(&ctx, &values(cols, 11), Shape::new(vec![cols]));
        let factors = factors(&ctx, cols);
        for operand in 0..6 {
            let mut operands = [
                input.clone(),
                input.clone(),
                vector.clone(),
                vector.clone(),
                vector.clone(),
                factors.clone(),
            ];
            let t = &operands[operand];
            let bytes = raw(t);
            let offset = if operand == 5 { 4 } else { 2 };
            let storage = MusaBuffer::alloc_on(&ctx, bytes.len() + offset).unwrap();
            let view = storage.view(offset, bytes.len()).unwrap();
            view.copy_from_host(&bytes).unwrap();
            operands[operand] = view.as_tensor(t.shape().clone(), t.dtype()).unwrap();
            let [x, r, w, b, nb, inv] = &operands;
            let split = if cols == 1152 {
                crate::kernels::fused::bias_residual_layer_bf16(&ctx, x, Some(b), r, w, nb, 1e-6)
            } else {
                crate::kernels::fused::bias_residual_rms_bf16(&ctx, x, Some(b), r, w, 1e-6)
            }
            .unwrap();
            let expected = quantize_w8a8_activation(&ctx, &split.normalized, Some(inv)).unwrap();
            let (hidden, actual) = norm_quantize_w8a8_activation(
                &ctx,
                x,
                Some((r, Some(b))),
                w,
                (cols == 1152).then_some(nb),
                1e-6,
                Some(inv),
            )
            .unwrap();
            ctx.synchronize().unwrap();
            compare(&actual, &expected);
            assert_eq!(raw(&hidden.unwrap()), raw(&split.hidden));
        }
        let short = upload(&ctx, &[0.0], Shape::new(vec![1]));
        let wrong = zeros_as(&ctx, Shape::new(vec![cols]), DType::BF16);
        assert!(
            norm_quantize_w8a8_activation(&ctx, &input, None, &short, None, 1e-6, None).is_err()
        );
        assert!(norm_quantize_w8a8_activation(
            &ctx,
            &input,
            None,
            &vector,
            Some(&short),
            1e-6,
            None
        )
        .is_err());
        assert!(norm_quantize_w8a8_activation(
            &ctx,
            &input,
            None,
            &vector,
            None,
            1e-6,
            Some(&wrong)
        )
        .is_err());
        assert!(norm_quantize_w8a8_activation(
            &ctx,
            &input,
            Some((&vector, None)),
            &vector,
            None,
            1e-6,
            None
        )
        .is_err());
    }
}
