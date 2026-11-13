//! BF16 text/image regression runner. Inputs are JSON exported from repository fixtures.
use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{GeneralQwen3VL, ImageInput, LlmInput, LlmTrait};
use serde::Deserialize;
use std::{path::Path, time::Instant};

#[derive(Deserialize)]
struct Fixture {
    tokens: Vec<u32>,
    greedy_tokens: Vec<u32>,
    pixel_shape: Option<Vec<usize>>,
    pixels: Option<Vec<f32>>,
    grids: Option<Vec<[u32; 3]>>,
}
fn values(t: &Tensor) -> Vec<f32> {
    match t.dtype() {
        DType::BF16 => t.as_bf16().unwrap().iter().map(|v| v.to_f32()).collect(),
        DType::F32 => t.as_f32().unwrap().to_vec(),
        dtype => panic!("unexpected output dtype: {dtype}"),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: qwen3vl_musa_check MODEL FIXTURE.json OUTPUT.json".into());
    }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let start = Instant::now();
    let mut model = GeneralQwen3VL::from_dir(Path::new(&args[1]), Device::Musa(0))?;
    model.backend().synchronize()?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pixels = match (&fixture.pixel_shape, &fixture.pixels) {
        (Some(shape), Some(data)) => Some(Tensor::from_bf16(
            shape.clone(),
            &data
                .iter()
                .map(|v| half::bf16::from_f32(*v))
                .collect::<Vec<_>>(),
        )?),
        (None, None) => None,
        _ => return Err("pixel shape/data must both be present".into()),
    };
    let start = Instant::now();
    let mut logits = match &pixels {
        Some(p) => model.prefill(LlmInput::with_image(
            &fixture.tokens,
            ImageInput::new(p, fixture.grids.as_deref().ok_or("missing grids")?),
        ))?,
        None => model.prefill(LlmInput::text(&fixture.tokens))?,
    };
    model.backend().synchronize()?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let host = model.backend().to_cpu(&logits)?;
    let all = values(&host);
    let vocab = model.vocab_size();
    let first_logits = all[all.len() - vocab..].to_vec();
    let mut generated = Vec::new();
    let mut decode_ms = Vec::new();
    for step in 0..fixture.greedy_tokens.len() {
        model.backend().synchronize()?;
        let host = model.backend().to_cpu(&logits)?;
        let all = values(&host);
        let row = &all[all.len() - vocab..];
        if row.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        let token =
            row.iter()
                .enumerate()
                .fold(0, |best, (i, v)| if *v > row[best] { i } else { best }) as u32;
        generated.push(token);
        if step + 1 < fixture.greedy_tokens.len() {
            let start = Instant::now();
            logits = model.forward(&[token], (fixture.tokens.len() + step) as u32)?;
            model.backend().synchronize()?;
            decode_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let matched = generated == fixture.greedy_tokens;
    let result = serde_json::json!({"device":"musa:0", "load_ms":load_ms, "prefill_ms":prefill_ms, "decode_ms":decode_ms, "logits_last":first_logits, "generated":generated, "expected":fixture.greedy_tokens, "greedy_match":matched});
    std::fs::write(&args[3], serde_json::to_vec(&result)?)?;
    println!("load={load_ms:.2}ms prefill={prefill_ms:.2}ms greedy_match={matched} generated={generated:?}");
    if !matched {
        return Err("greedy tokens differ from repository reference".into());
    }
    Ok(())
}
