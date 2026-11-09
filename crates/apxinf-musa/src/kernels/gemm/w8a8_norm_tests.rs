use super::*;
use crate::kernels::test_support::{bf16_bytes, read_back, upload, values, zeros_as};
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

#[test]
#[ignore = "needs a MUSA device"]
fn action_norm_quant_matches_split_and_validates_views() {
    let ctx = MusaContext::new(0).unwrap();
    for (rows, cols) in [(1, 1024), (10, 1024), (32, 1024), (3, 150), (3, 2048)] {
        for case in 0..3 {
            let mut data = values(rows * cols, 52);
            if case == 0 {
                data.fill(0.0);
            }
            if case == 1 {
                data[0] = 128.0;
                data[1] = -128.0;
            }
            let input = upload(&ctx, &data, Shape::new(vec![rows, cols]));
            let residual = upload(&ctx, &values(rows * cols, 25), Shape::new(vec![rows, cols]));
            let gate = upload(&ctx, &values(cols * 3, 26), Shape::new(vec![cols * 3]));
            let style = upload(&ctx, &values(cols * 3, 27), Shape::new(vec![cols * 3]));
            let factors = factors(&ctx, cols);
            for inv in [None, Some(&factors)] {
                let normalized =
                    crate::kernels::norm::adaptive_rms_bf16(&ctx, &input, &style, 1e-6).unwrap();
                let expected = quantize_w8a8_activation(&ctx, &normalized, inv).unwrap();
                let actual =
                    adaptive_rms_quantize_w8a8_activation(&ctx, &input, &style, 1e-6, inv).unwrap();
                ctx.synchronize().unwrap();
                compare(&actual, &expected);
                let split = crate::kernels::fused::adaptive_gate_residual_rms_bf16(
                    &ctx, &input, &residual, &gate, &style, 1e-6,
                )
                .unwrap();
                let expected = quantize_w8a8_activation(&ctx, &split.normalized, inv).unwrap();
                let (hidden, actual) = adaptive_gate_residual_rms_quantize_w8a8_activation(
                    &ctx, &input, &residual, &gate, &style, 1e-6, inv,
                )
                .unwrap();
                ctx.synchronize().unwrap();
                compare(&actual, &expected);
                assert_eq!(read_back(&hidden), read_back(&split.hidden));
            }
        }
    }
    let cols = 1024;
    let input = upload(&ctx, &values(3 * cols, 10), Shape::new(vec![3, cols]));
    let residual = input.clone();
    let style = upload(&ctx, &values(3 * cols, 12), Shape::new(vec![3 * cols]));
    let inv = factors(&ctx, cols);
    // Individually misalign each vector operand; compare the exact fallback.
    for operand in 0..5 {
        let mut operands = [
            input.clone(),
            residual.clone(),
            style.clone(),
            style.clone(),
            inv.clone(),
        ];
        let t = &operands[operand];
        let size = t.size_in_bytes();
        let offset = if operand == 4 { 4 } else { 2 };
        let mut bytes = vec![0; size];
        MusaBuffer::from_tensor(t)
            .unwrap()
            .copy_to_host(&mut bytes)
            .unwrap();
        let storage = MusaBuffer::alloc_on(&ctx, size + offset).unwrap();
        let view = storage.view(offset, size).unwrap();
        view.copy_from_host(&bytes).unwrap();
        operands[operand] = view.as_tensor(t.shape().clone(), t.dtype()).unwrap();
        let [input, residual, gate, style, inv] = &operands;
        let split = crate::kernels::fused::adaptive_gate_residual_rms_bf16(
            &ctx, input, residual, gate, style, 1e-6,
        )
        .unwrap();
        let expected = quantize_w8a8_activation(&ctx, &split.normalized, Some(inv)).unwrap();
        let (hidden, actual) = adaptive_gate_residual_rms_quantize_w8a8_activation(
            &ctx,
            input,
            residual,
            gate,
            style,
            1e-6,
            Some(inv),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        compare(&actual, &expected);
        assert_eq!(read_back(&hidden), read_back(&split.hidden));
    }
    let wrong_style = upload(&ctx, &[0.0], Shape::new(vec![1]));
    assert!(adaptive_rms_quantize_w8a8_activation(&ctx, &input, &wrong_style, 1e-6, None).is_err());
    let wrong_factor = zeros_as(&ctx, Shape::new(vec![cols]), DType::BF16);
    assert!(
        adaptive_rms_quantize_w8a8_activation(&ctx, &input, &style, 1e-6, Some(&wrong_factor))
            .is_err()
    );
    let short_residual = upload(&ctx, &values(cols, 4), Shape::new(vec![1, cols]));
    assert!(adaptive_gate_residual_rms_quantize_w8a8_activation(
        &ctx,
        &input,
        &short_residual,
        &style,
        &style,
        1e-6,
        None
    )
    .is_err());
}

#[test]
#[ignore = "needs MP22"]
fn action_norm_quant_graph_replays_changed_inputs() {
    let ctx = Arc::new(MusaContext::new(0).unwrap());
    let (rows, cols) = (10, 1024);
    let data = values(rows * cols, 13);
    let input = upload(&ctx, &data, Shape::new(vec![rows, cols]));
    let residual = upload(&ctx, &values(rows * cols, 14), Shape::new(vec![rows, cols]));
    let gate = upload(&ctx, &values(3 * cols, 15), Shape::new(vec![3 * cols]));
    let style = upload(&ctx, &values(3 * cols, 16), Shape::new(vec![3 * cols]));
    let factors = factors(&ctx, cols);
    for inv in [None, Some(&factors)] {
        for gated in [false, true] {
            let a_workspace = crate::workspace::GraphWorkspace::new(131072, 0).unwrap();
            let b_workspace = crate::workspace::GraphWorkspace::new(131072, 0).unwrap();
            let (a_graph, (expected_hidden, expected)) = crate::graph::capture(ctx.clone(), || {
                crate::workspace::with_workspace(&a_workspace, || {
                    let (hidden, normalized) = if gated {
                        let out = crate::kernels::fused::adaptive_gate_residual_rms_bf16(
                            &ctx, &input, &residual, &gate, &style, 1e-6,
                        )?;
                        (Some(out.hidden), out.normalized)
                    } else {
                        (
                            None,
                            crate::kernels::norm::adaptive_rms_bf16(&ctx, &input, &style, 1e-6)?,
                        )
                    };
                    Ok((hidden, quantize_w8a8_activation(&ctx, &normalized, inv)?))
                })
            })
            .unwrap();
            let (b_graph, (hidden, actual)) = crate::graph::capture(ctx.clone(), || {
                crate::workspace::with_workspace(&b_workspace, || {
                    if gated {
                        let (h, q) = adaptive_gate_residual_rms_quantize_w8a8_activation(
                            &ctx, &input, &residual, &gate, &style, 1e-6, inv,
                        )?;
                        Ok((Some(h), q))
                    } else {
                        Ok((
                            None,
                            adaptive_rms_quantize_w8a8_activation(&ctx, &input, &style, 1e-6, inv)?,
                        ))
                    }
                })
            })
            .unwrap();
            for iteration in 0..24 {
                let sign = [1.0, -1.0, 0.0, -1.0][iteration % 4];
                let input_values: Vec<f32> = data.iter().map(|v| v * sign).collect();
                MusaBuffer::from_tensor(&input)
                    .unwrap()
                    .copy_from_host(&bf16_bytes(&input_values))
                    .unwrap();
                for (tensor, seed) in [(&style, iteration + 50), (&gate, iteration + 90)] {
                    MusaBuffer::from_tensor(tensor)
                        .unwrap()
                        .copy_from_host(&bf16_bytes(&values(3 * cols, seed as u32)))
                        .unwrap();
                }
                ctx.synchronize().unwrap();
                a_graph.replay().unwrap();
                b_graph.replay().unwrap();
                ctx.synchronize().unwrap();
                compare(&actual, &expected);
                if let (Some(a), Some(b)) = (&hidden, &expected_hidden) {
                    assert_eq!(read_back(a), read_back(b));
                }
            }
        }
    }
}
