//! Fixed-input GR00T N1.7 MUSA model-core benchmark.
//!
//! The input directory is produced by NVIDIA's official processor. Timing
//! begins with those preprocessed host tensors and ends after action D2H.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, ExecutionPolicy, LoadOptions, ModelPrecision, Observation, VisionObservation,
    VlaMetadata, VlaRequest,
};
use half::bf16;
use serde_json::Value;

const FIXTURE_SCHEMA: &str = "apxinf.gr00t-n1.7.preprocessed-fixture.v1";
const DEFAULT_EXECUTION_POLICY: ExecutionPolicy = ExecutionPolicy::RequireGraph;

struct Arguments {
    checkpoint: PathBuf,
    backbone: PathBuf,
    fixture: PathBuf,
    precision: ModelPrecision,
    device: usize,
    warmup: usize,
    iterations: usize,
    calibration: Option<PathBuf>,
    tactics: Option<PathBuf>,
    output: Option<PathBuf>,
    autotune: bool,
    execution_policy: ExecutionPolicy,
}

struct Fixture {
    pixel_values: Tensor,
    grid: Vec<[u32; 3]>,
    token_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    state: Tensor,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

struct RequestInput {
    observation: Observation,
    attention_mask: Vec<u8>,
    grid: Vec<[u32; 3]>,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

impl RequestInput {
    fn from_fixture(fixture: Fixture) -> Self {
        let Fixture {
            pixel_values,
            grid,
            token_ids,
            attention_mask,
            state,
            noise,
            embodiment_id,
            name,
        } = fixture;
        Self {
            observation: Observation {
                vision: VisionObservation::Patches(pixel_values),
                token_ids,
                state: Some(state),
                action_mask: None,
            },
            attention_mask,
            grid,
            noise,
            embodiment_id,
            name,
        }
    }

    fn request(&self) -> VlaRequest<'_> {
        let metadata = VlaMetadata {
            attention_mask: Some(&self.attention_mask),
            image_grid_thw: Some(&self.grid),
            embodiment_id: Some(self.embodiment_id),
            planning: None,
        };
        VlaRequest::provided_with_metadata(&self.observation, &self.noise, metadata)
    }
}

struct TensorEntry {
    path: PathBuf,
    shape: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments()?;
    let input = RequestInput::from_fixture(load_fixture(&args.fixture)?);
    let mut options = LoadOptions {
        model_name: Some("gr00t".into()),
        precision: args.precision,
        calibration_path: args.calibration.clone(),
        tuning_path: args.tactics.clone(),
        autotune: args.autotune,
        execution_policy: Some(args.execution_policy),
        ..LoadOptions::default()
    };
    options
        .assets
        .insert("backbone".into(), args.backbone.clone());

    let load_started = Instant::now();
    let model = AutoModel::load_model(Device::Musa(args.device), &args.checkpoint, &options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;

    for _ in 0..args.warmup {
        let request = input.request();
        black_box(model.infer_host_f32(&request)?);
    }
    let execution = model.vla()?.execution_mode();
    let execution_matches_policy = match args.execution_policy {
        ExecutionPolicy::Eager => execution == "no_graph",
        ExecutionPolicy::PreferGraph => matches!(execution, "musa-graph" | "no_graph"),
        ExecutionPolicy::RequireGraph => execution == "musa-graph",
    };
    if !execution_matches_policy {
        return Err(format!(
            "GR00T MUSA benchmark requested {}, runtime reported {execution}",
            execution_policy_name(args.execution_policy),
        )
        .into());
    }

    let mut samples = Vec::with_capacity(args.iterations);
    let mut last_output = Vec::new();
    let mut sample_details = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let request = input.request();
        let started = Instant::now();
        let output = model.infer_host_f32(&request)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        samples.push(elapsed_ms);
        last_output = output.clone();
        sample_details.push(serde_json::json!({
            "iteration": iteration,
            "fixture": input.name,
            "token_count": input.observation.token_ids.len(),
            "latency_ms": elapsed_ms,
        }));
        black_box(&last_output);
    }
    let summary = latency_summary(&samples)?;
    let [horizon, action_dim] = model.vla()?.action_shape();
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.benchmark.v2",
        "mode": "fixed",
        "fixture": &input.name,
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "device": args.device,
        "precision": precision_name(args.precision),
        "requested_execution": execution_policy_name(args.execution_policy),
        "execution": execution,
        "timing_boundary": "official-processor tensors through synchronized model core and action D2H",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_ms": load_ms,
        "latency_ms": summary,
        "samples_detail": sample_details,
        "input": {
            "pixel_values": observation_shape(&input.observation),
            "image_grid_thw": &input.grid,
            "token_count": input.observation.token_ids.len(),
            "state": input.observation.state.as_ref().map(|value| value.shape().dims()),
            "embodiment_id": input.embodiment_id,
        },
        "output": {
            "shape": [horizon, action_dim],
            "sum": last_output.iter().copied().map(f64::from).sum::<f64>(),
            "head": last_output.iter().take(16).copied().collect::<Vec<_>>(),
            "values": last_output,
        },
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        std::fs::write(&path, format!("{rendered}\n"))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn observation_shape(observation: &Observation) -> &[usize] {
    match &observation.vision {
        VisionObservation::Patches(value) => value.shape().dims(),
        VisionObservation::RgbU8 { .. } => &[],
    }
}

fn parse_arguments() -> Result<Arguments, Box<dyn std::error::Error>> {
    let values = std::env::args().collect::<Vec<_>>();
    if values.len() < 5 {
        return Err(format!(
            "usage: {} <checkpoint> <backbone> <fixture> <bf16|fp8|int8> [device=0] [warmup=10] [iterations=50] [calibration|-] [tactics|-] [output|-] [--autotune] [--execution <eager|prefer-graph|require-graph>]",
            values.first().map(String::as_str).unwrap_or("gr00t_bench_musa")
        )
        .into());
    }
    let integer = |index: usize, default: usize, label: &str| {
        values
            .get(index)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {label} {value:?}: {error}"))
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let optional_path = |index: usize| {
        values
            .get(index)
            .filter(|value| value.as_str() != "-")
            .map(PathBuf::from)
    };
    let precision = match values[4].as_str() {
        "bf16" => ModelPrecision::Bf16,
        "fp8" => ModelPrecision::Fp8,
        "int8" => ModelPrecision::W8A8,
        value => return Err(format!("invalid precision {value:?}; expected bf16|fp8|int8").into()),
    };
    let iterations = integer(7, 50, "iteration count")?;
    if iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    let mut autotune = false;
    let mut execution_policy = DEFAULT_EXECUTION_POLICY;
    let mut execution_was_set = false;
    let mut trailing = values.iter().skip(11);
    while let Some(value) = trailing.next() {
        match value.as_str() {
            "--autotune" => {
                if autotune {
                    return Err("--autotune was specified more than once".into());
                }
                autotune = true;
            }
            "--execution" => {
                if execution_was_set {
                    return Err("--execution was specified more than once".into());
                }
                let value = trailing.next().ok_or("--execution requires a value")?;
                execution_policy = parse_execution_policy(value)?;
                execution_was_set = true;
            }
            _ => {
                return Err(format!(
                    "invalid trailing argument {value:?}; expected --autotune or --execution"
                )
                .into());
            }
        }
    }
    Ok(Arguments {
        checkpoint: PathBuf::from(&values[1]),
        backbone: PathBuf::from(&values[2]),
        fixture: PathBuf::from(&values[3]),
        precision,
        device: integer(5, 0, "MUSA device")?,
        warmup: integer(6, 10, "warmup count")?,
        iterations,
        calibration: optional_path(8),
        tactics: optional_path(9),
        output: optional_path(10),
        autotune,
        execution_policy,
    })
}

fn parse_execution_policy(value: &str) -> Result<ExecutionPolicy, Box<dyn std::error::Error>> {
    match value {
        "eager" => Ok(ExecutionPolicy::Eager),
        "prefer-graph" => Ok(ExecutionPolicy::PreferGraph),
        "require-graph" => Ok(ExecutionPolicy::RequireGraph),
        _ => Err(format!(
            "invalid execution policy {value:?}; expected eager|prefer-graph|require-graph"
        )
        .into()),
    }
}

fn execution_policy_name(policy: ExecutionPolicy) -> &'static str {
    match policy {
        ExecutionPolicy::Eager => "eager",
        ExecutionPolicy::PreferGraph => "prefer-graph",
        ExecutionPolicy::RequireGraph => "require-graph",
    }
}

fn load_fixture(root: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let schema = string(&manifest, "schema")?;
    if schema != FIXTURE_SCHEMA {
        return Err(format!("unsupported fixture schema {schema:?}").into());
    }
    let pixels = entry(root, &manifest, "pixel_values", "bfloat16")?;
    let grid = entry(root, &manifest, "image_grid_thw", "uint32")?;
    let tokens = entry(root, &manifest, "token_ids", "uint32")?;
    let mask = entry(root, &manifest, "attention_mask", "uint8")?;
    let state = entry(root, &manifest, "state", "bfloat16")?;
    let noise = entry(root, &manifest, "noise", "bfloat16")?;
    if grid.shape.len() != 2 || grid.shape[1] != 3 {
        return Err(format!("image_grid_thw must be [images, 3], got {:?}", grid.shape).into());
    }
    let grid = read_u32(&grid)?
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    Ok(Fixture {
        pixel_values: read_bf16(&pixels)?,
        grid,
        token_ids: read_u32(&tokens)?,
        attention_mask: read_u8(&mask)?,
        state: read_bf16(&state)?,
        noise: read_bf16(&noise)?,
        embodiment_id: usize::try_from(
            manifest
                .get("embodiment_id")
                .and_then(Value::as_u64)
                .ok_or("fixture embodiment_id must be an integer")?,
        )?,
        name: string(&manifest, "fixture")?.to_owned(),
    })
}

fn entry(
    root: &Path,
    manifest: &Value,
    name: &str,
    dtype: &str,
) -> Result<TensorEntry, Box<dyn std::error::Error>> {
    let value = manifest
        .get("tensors")
        .and_then(|tensors| tensors.get(name))
        .ok_or_else(|| format!("fixture is missing tensors.{name}"))?;
    if string(value, "dtype")? != dtype {
        return Err(format!("fixture tensor {name} must have dtype {dtype}").into());
    }
    let shape = value
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("fixture tensor {name} has no shape"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| format!("invalid {name} shape"))
        })
        .map(|value| {
            value.and_then(|value| usize::try_from(value).map_err(|error| error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorEntry {
        path: root.join(string(value, "file")?),
        shape,
    })
}

fn read_bf16(entry: &TensorEntry) -> Result<Tensor, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("fixture tensor byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            bytes.len(),
            entry.shape
        )
        .into());
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    Ok(Tensor::from_bf16(entry.shape.clone(), &values)?)
}

fn read_u32(entry: &TensorEntry) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or("fixture tensor byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            bytes.len(),
            entry.shape
        )
        .into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok(values)
}

fn read_u8(entry: &TensorEntry) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let values = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?;
    if values.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            values.len(),
            entry.shape
        )
        .into());
    }
    Ok(values)
}

fn elements(shape: &[usize]) -> Result<usize, Box<dyn std::error::Error>> {
    shape
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .ok_or_else(|| "fixture tensor size overflow".into())
}

fn latency_summary(samples: &[f64]) -> Result<Value, Box<dyn std::error::Error>> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return Err("latency sample list is empty".into());
    }
    Ok(serde_json::json!({
        "samples": samples,
        "p50": percentile(&sorted, 0.50),
        "p95": percentile(&sorted, 0.95),
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "min": sorted[0],
        "max": sorted[sorted.len() - 1],
    }))
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = ((sorted.len() as f64) * quantile).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn precision_name(precision: ModelPrecision) -> &'static str {
    match precision {
        ModelPrecision::Auto => "auto",
        ModelPrecision::Bf16 => "bf16",
        ModelPrecision::Fp8 => "fp8",
        ModelPrecision::W8A8 => "int8",
    }
}

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("fixture field {field:?} must be a string").into())
}

#[cfg(test)]
mod tests {
    use super::{
        execution_policy_name, parse_execution_policy, ExecutionPolicy, DEFAULT_EXECUTION_POLICY,
    };

    #[test]
    fn execution_policy_defaults_to_strict_graph_and_parses_all_modes() {
        assert_eq!(DEFAULT_EXECUTION_POLICY, ExecutionPolicy::RequireGraph);
        for (name, expected) in [
            ("eager", ExecutionPolicy::Eager),
            ("prefer-graph", ExecutionPolicy::PreferGraph),
            ("require-graph", ExecutionPolicy::RequireGraph),
        ] {
            assert_eq!(parse_execution_policy(name).unwrap(), expected);
            assert_eq!(execution_policy_name(expected), name);
        }
        assert!(parse_execution_policy("graph").is_err());
    }
}
