#![cfg(feature = "musa")]
use apxinf_model::musa_qwen3vl::{validate_decode_transition, NativeSequenceState};

#[test]
fn native_state_rejects_empty_prefill() {
    let mut state = NativeSequenceState::default();
    assert!(validate_decode_transition(&mut state, 0, 0).is_err());
}

#[test]
fn native_state_advances_prefill_then_one_token() {
    let mut state = NativeSequenceState::default();
    validate_decode_transition(&mut state, 13, 0).unwrap();
    assert_eq!(state.position(), 13);
    validate_decode_transition(&mut state, 1, 13).unwrap();
    assert_eq!(state.position(), 14);
}

#[test]
fn native_state_rejects_decode_gap() {
    let mut state = NativeSequenceState::default();
    validate_decode_transition(&mut state, 13, 0).unwrap();
    assert!(validate_decode_transition(&mut state, 1, 15).is_err());
}

#[test]
fn native_state_reset_allows_new_prefill() {
    let mut state = NativeSequenceState::default();
    validate_decode_transition(&mut state, 13, 0).unwrap();
    state.reset();
    validate_decode_transition(&mut state, 7, 0).unwrap();
    assert_eq!(state.position(), 7);
}

/// Fixture: exported Qwen image input and four little-endian float32 HF outputs.
#[test]
#[ignore = "requires M1000, a BF16 checkpoint and exported vision fixtures"]
fn native_vision_checkpoint_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_core::{Device, Tensor};
    use apxinf_model::musa_qwen3vl::Qwen3VlMusaExecutor;
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Input {
        pixels: Vec<f32>,
        pixel_shape: Vec<usize>,
        grids: Vec<[u32; 3]>,
    }
    let dir = PathBuf::from(env::var("QWEN_VISION_FIXTURE_DIR")?);
    let input: Input = serde_json::from_slice(&fs::read(dir.join("input.json"))?)?;
    let pixels = Tensor::from_bf16(
        input.pixel_shape,
        &input
            .pixels
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    let model =
        Qwen3VlMusaExecutor::load(0, &PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?))?;
    assert!(model.encode_image(&pixels, &[]).is_err());
    let out = model.encode_image(&pixels, &input.grids)?;
    model.synchronize()?;
    assert_eq!(out.deepstack.len(), 3);
    for (name, t) in std::iter::once(("vision_primary".to_owned(), out.primary)).chain(
        out.deepstack
            .into_iter()
            .enumerate()
            .map(|(i, t)| (format!("vision_deepstack_{i}"), t)),
    ) {
        assert_eq!(t.device(), Device::Musa(0));
        assert_eq!(t.shape().dims(), [100, 2048]);
        let actual = apxinf_musa::transfers::to_cpu(&t)?.to_f32_vec()?;
        let bytes = fs::read(dir.join(format!("{name}.f32")))?;
        assert_eq!(bytes.len(), actual.len() * 4);
        let (mut dot, mut aa, mut bb, mut squared) = (0f64, 0f64, 0f64, 0f64);
        for (v, b) in actual.into_iter().zip(bytes.chunks_exact(4)) {
            let v = v as f64;
            let w = f32::from_le_bytes(b.try_into()?) as f64;
            dot += v * w;
            aa += v * v;
            bb += w * w;
            squared += (v - w) * (v - w);
        }
        let cosine = dot / (aa * bb).sqrt();
        let rel = (squared / bb).sqrt();
        eprintln!("{name}: cosine={cosine:.8} relative_l2={rel:.6}");
        assert!(
            cosine > 0.99 && rel < 0.15,
            "{name}: cosine={cosine}, relative_l2={rel}"
        );
    }
    Ok(())
}

/// Migration oracle, not an independent HF reference: compare all BF16 bits
/// against the existing executor, for both native and online SDPA shapes.
#[test]
#[ignore = "requires M1000 and Qwen image fixture"]
fn native_vision_matches_existing_executor() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_core::{Device, Tensor};
    use apxinf_model::{musa_qwen3vl::Qwen3VlMusaExecutor, GeneralQwen3VL};
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Input {
        pixels: Vec<f32>,
    }
    let dir = PathBuf::from(env::var("QWEN_VISION_FIXTURE_DIR")?);
    let checkpoint = PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?);
    let input: Input = serde_json::from_slice(&fs::read(dir.join("input.json"))?)?;
    let cases: Vec<_> = [20usize, 8]
        .iter()
        .map(|&side| {
            let vals: Vec<_> = input.pixels[..side * side * 1536]
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect();
            Ok((
                Tensor::from_bf16(vec![side * side, 1536], &vals)?,
                [[1, side as u32, side as u32]],
            ))
        })
        .collect::<apxinf_core::Result<_>>()?;
    let collect = |primary: Tensor, stack: Vec<Tensor>| -> apxinf_core::Result<Vec<Vec<u16>>> {
        std::iter::once(primary)
            .chain(stack)
            .map(|t| {
                Ok(apxinf_musa::transfers::to_cpu(&t)?
                    .as_bf16()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect())
            })
            .collect()
    };
    let mut reference = Vec::new();
    {
        let model = GeneralQwen3VL::from_dir(&checkpoint, Device::Musa(0))?;
        for (p, g) in &cases {
            let out = model.forward_vision(&model.backend().to_device(p)?, g)?;
            model.backend().synchronize()?;
            reference.push(collect(out.primary, out.deepstack)?);
        }
    }
    let model = Qwen3VlMusaExecutor::load(0, &checkpoint)?;
    let mut errors = 0;
    for ((p, g), old) in cases.iter().zip(reference) {
        let out = model.encode_image(p, g)?;
        model.synchronize()?;
        let new = collect(out.primary, out.deepstack)?;
        assert_eq!(new.len(), old.len());
        for (i, (a, b)) in new.iter().zip(old).enumerate() {
            assert_eq!(a.len(), b.len());
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            eprintln!("grid={g:?} output={i} different={diff}/{}", a.len());
            errors += diff;
        }
    }
    assert_eq!(errors, 0, "vision migration changed BF16 outputs");
    Ok(())
}

#[test]
#[ignore = "requires M1000 and exported HF text and image fixtures"]
fn native_generation_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_core::Tensor;
    use apxinf_model::musa_qwen3vl::Qwen3VlMusaExecutor;
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Case {
        tokens: Vec<u32>,
        greedy_tokens: Vec<u32>,
        pixels: Option<Vec<f32>>,
        pixel_shape: Option<Vec<usize>>,
        grids: Option<Vec<[u32; 3]>>,
    }
    let dir = PathBuf::from(env::var("QWEN_GENERATION_FIXTURES")?);
    let mut model =
        Qwen3VlMusaExecutor::load(0, &PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?))?;
    for mode in ["text", "image"] {
        model.reset()?;
        assert!(model.decode_one(1).is_err());
        assert!(model.prefill_text(&[]).is_err());
        let case: Case = serde_json::from_slice(&fs::read(dir.join(format!("{mode}.json")))?)?;
        let mut logits = match case.pixels {
            None => model.prefill_text(&case.tokens)?,
            Some(p) => {
                let pixels = Tensor::from_bf16(
                    case.pixel_shape.unwrap(),
                    &p.into_iter().map(half::bf16::from_f32).collect::<Vec<_>>(),
                )?;
                model.prefill_image(&case.tokens, &pixels, case.grids.as_deref().unwrap())?
            }
        };
        assert!(model.prefill_text(&[1]).is_err());
        assert!(model.decode_one(u32::MAX).is_err());
        model.synchronize()?;
        let actual = apxinf_musa::transfers::to_cpu(&logits)?.to_f32_vec()?;
        let reference = fs::read(dir.join(format!("{mode}.logits")))?;
        assert_eq!(actual.len() * 4, reference.len());
        let expected: Vec<f32> = reference
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let top = |s: &[f32]| {
            let mut ids: Vec<usize> = (0..s.len()).collect();
            ids.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            ids.truncate(5);
            ids
        };
        let a = top(&actual);
        let b = top(&expected);
        assert_eq!(a[0], b[0]);
        assert!(a.iter().all(|v| b.contains(v)));
        let (mut dot, mut aa, mut bb) = (0., 0., 0.);
        for (&v, &w) in actual.iter().zip(&expected) {
            let (v, w) = (v as f64, w as f64);
            dot += v * w;
            aa += v * v;
            bb += w * w;
        }
        let cos = dot / (aa * bb).sqrt();
        assert!(cos > 0.999);
        let mut generated = vec![];
        for i in 0..case.greedy_tokens.len() {
            let token = model.greedy_token(&logits)?;
            generated.push(token);
            if i + 1 < case.greedy_tokens.len() {
                logits = model.decode_one(token)?;
            }
        }
        eprintln!("{mode} cosine={cos:.8} top5=5/5 tokens={generated:?}");
        assert_eq!(generated, case.greedy_tokens);
    }
    Ok(())
}

/// Compare identical prefixes rather than allowing two samplers to diverge.
#[test]
#[ignore = "requires M1000 and the exported 2B text fixture"]
fn native_cached_prefix_matches_recompute() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_model::musa_qwen3vl::Qwen3VlMusaExecutor;
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Text {
        tokens: Vec<u32>,
        greedy_tokens: Vec<u32>,
    }
    let dir = PathBuf::from(env::var("QWEN_GENERATION_FIXTURES")?);
    let case: Text = serde_json::from_slice(&fs::read(dir.join("text.json"))?)?;
    assert_eq!((case.tokens.len(), case.greedy_tokens.len()), (13, 10));
    let mut model =
        Qwen3VlMusaExecutor::load(0, &PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?))?;
    let _ = model.prefill_text(&case.tokens)?;
    let mut cached = Vec::new();
    for &id in &case.greedy_tokens[..9] {
        let t = model.decode_one(id)?;
        model.synchronize()?;
        cached.push(apxinf_musa::transfers::to_cpu(&t)?.to_f32_vec()?);
    }
    let mut prefix = case.tokens;
    for (i, a) in cached.iter().enumerate() {
        prefix.push(case.greedy_tokens[i]);
        model.reset()?;
        let t = model.prefill_text(&prefix)?;
        model.synchronize()?;
        let b = apxinf_musa::transfers::to_cpu(&t)?.to_f32_vec()?;
        assert_eq!(a.len(), b.len());
        let (mut dot, mut aa, mut bb, mut max) = (0f64, 0f64, 0f64, 0f64);
        let (mut ia, mut ib) = (0, 0);
        for (j, (&v, &w)) in a.iter().zip(&b).enumerate() {
            assert!(v.is_finite() && w.is_finite());
            if v > a[ia] {
                ia = j
            };
            if w > b[ib] {
                ib = j
            };
            let (v, w) = (v as f64, w as f64);
            dot += v * w;
            aa += v * v;
            bb += w * w;
            max = max.max((v - w).abs());
        }
        let cosine = dot / (aa * bb).sqrt();
        eprintln!(
            "prefix={} cosine={cosine:.9} max_abs={max:.6} cached_top={ia} full_top={ib}",
            prefix.len()
        );
        assert_eq!(ia, ib);
        assert_eq!(ia as u32, case.greedy_tokens[i + 1]);
        assert!(cosine > 0.999);
    }
    Ok(())
}

/// Size-specific image fixture and saved generic-MUSA logits (not an HF logits oracle).
#[test]
#[ignore = "requires M1000, checkpoint, and matching saved baseline"]
fn native_image_matches_saved_baseline() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_core::Tensor;
    use apxinf_model::musa_qwen3vl::Qwen3VlMusaExecutor;
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Input {
        tokens: Vec<u32>,
        greedy_tokens: Vec<u32>,
        pixels: Vec<f32>,
        pixel_shape: Vec<usize>,
        grids: Vec<[u32; 3]>,
    }
    #[derive(serde::Deserialize)]
    struct Baseline {
        logits_last: Vec<f32>,
        generated: Vec<u32>,
    }
    let case: Input = serde_json::from_slice(&fs::read(env::var("QWEN_IMAGE_INPUT")?)?)?;
    let base: Baseline = serde_json::from_slice(&fs::read(env::var("QWEN_IMAGE_BASELINE")?)?)?;
    assert_eq!(case.greedy_tokens.len(), 10);
    assert_eq!(base.generated, case.greedy_tokens);
    let mut model =
        Qwen3VlMusaExecutor::load(0, &PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?))?;
    let pixels = Tensor::from_bf16(
        case.pixel_shape,
        &case
            .pixels
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    let mut logits = model.prefill_image(&case.tokens, &pixels, &case.grids)?;
    model.synchronize()?;
    let actual = apxinf_musa::transfers::to_cpu(&logits)?.to_f32_vec()?;
    assert_eq!(actual.len(), base.logits_last.len());
    let (mut aa, mut bb, mut ab, mut max) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in actual.iter().zip(&base.logits_last) {
        assert!(x.is_finite() && y.is_finite());
        let (x, y) = (x as f64, y as f64);
        aa += x * x;
        bb += y * y;
        ab += x * y;
        max = max.max((x - y).abs());
    }
    let cosine = ab / (aa * bb).sqrt();
    let mut generated = vec![];
    for i in 0..10 {
        let id = model.greedy_token(&logits)?;
        generated.push(id);
        if i != 9 {
            logits = model.decode_one(id)?;
        }
    }
    eprintln!("old_path_cosine={cosine:.9} max_abs={max:.6} generated={generated:?}");
    assert_eq!(generated, case.greedy_tokens);
    assert!(cosine > 0.999);
    Ok(())
}

#[test]
#[ignore = "requires M1000 and exported HF text fixture"]
fn native_prefill_reports_stable_workspace() -> Result<(), Box<dyn std::error::Error>> {
    use apxinf_model::musa_qwen3vl::Qwen3VlMusaExecutor;
    use std::{env, fs, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Text {
        tokens: Vec<u32>,
    }
    let dir = PathBuf::from(env::var("QWEN_GENERATION_FIXTURES")?);
    let case: Text = serde_json::from_slice(&fs::read(dir.join("text.json"))?)?;
    let mut model =
        Qwen3VlMusaExecutor::load(0, &PathBuf::from(env::var("APXINF_QWEN3VL_MODEL_DIR")?))?;
    let first = model.prefill_text(&case.tokens)?;
    model.synchronize()?;
    let used = model.workspace_bytes_used();
    assert!(
        used > 0,
        "native prefill must use the persistent execution workspace"
    );
    let read = |t: &apxinf_core::Tensor| apxinf_musa::transfers::to_cpu(t)?.to_f32_vec();
    let original = read(&first)?;
    let mut changed = case.tokens.clone();
    *changed.last_mut().unwrap() = 1;
    model.reset()?;
    let second = model.prefill_text(&changed)?;
    model.synchronize()?;
    assert_eq!(model.workspace_bytes_used(), used);
    assert_ne!(original, read(&second)?);
    assert_eq!(original, read(&first)?);
    Ok(())
}
