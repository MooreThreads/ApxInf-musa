//! Load the MUSA GR00T configuration and establish the backend-owned runner.

use std::path::{Path, PathBuf};

use apxinf_core::{Device, Error, Result};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

use super::assets;
use super::model::executor::Gr00tPrecisionExecution;
use super::model::precision;
use super::weights::Gr00tWeights;
use super::{backend, Gr00tConfig, Gr00tModelRunner};

const BACKBONE_ASSET: &str = "backbone";

pub(crate) fn load(path: &Path, device: Device, options: &LoadOptions) -> Result<LoadedModel> {
    let Device::Musa(_) = device else {
        return Err(Error::Other(format!("GR00T requires MUSA, not {device}")));
    };
    validate_options(options)?;
    let root = artifact_root(path);
    if let Some(name) = options
        .assets
        .keys()
        .find(|name| name.as_str() != BACKBONE_ASSET)
    {
        return Err(Error::Other(format!(
            "GR00T does not recognize asset {name:?}; expected only {BACKBONE_ASSET:?}"
        )));
    }
    let backbone_path = assets::resolve_assets(
        root,
        options
            .assets
            .get(BACKBONE_ASSET)
            .map(|path| path.as_path()),
    )?;
    let config = Gr00tConfig::from_json_file(&root.join("config.json"))?;
    let backend = backend::create(device)?;
    let calibration = discover_calibration(root, options);
    let precision = resolve_precision(options.precision, &backend, calibration.as_deref())?;
    let (backbone_config, weights) = Gr00tWeights::from_safetensors(&config, &backbone_path, root)?;
    install_tactics(&backend, root, options)?;
    match precision {
        ModelPrecision::Fp8 => Ok(finish(
            Gr00tModelRunner::new(precision::build_fp8(
                root,
                &backbone_path,
                calibration
                    .as_deref()
                    .expect("resolve_precision requires a calibration profile"),
                config,
                backbone_config,
                weights,
                backend,
            )?),
            options,
        )),
        _ => Ok(finish(
            Gr00tModelRunner::new(precision::build(config, backbone_config, weights, backend)?),
            options,
        )),
    }
}

fn finish<E: Gr00tPrecisionExecution>(
    runner: Gr00tModelRunner<E>,
    options: &LoadOptions,
) -> LoadedModel {
    let runner = match options.execution_policy {
        Some(policy) => runner.with_execution_policy(policy),
        None => runner,
    };
    LoadedModel::Vla(Box::new(runner))
}

/// The explicit `calibration_path`, else `<model_dir>/calibration.json` when
/// it exists. The GR00T contract keeps calibration an external deployment
/// artifact, so a missing file is reported by `resolve_precision`, not created.
fn discover_calibration(root: &Path, options: &LoadOptions) -> Option<PathBuf> {
    if let Some(path) = &options.calibration_path {
        return Some(path.clone());
    }
    let default = root.join("calibration.json");
    default.is_file().then_some(default)
}

/// Resolve the requested precision against this device and build.
///
/// `auto` stays BF16 exactly like the CUDA family: a calibrated profile is an
/// explicit deployment choice, never a silent one. FP8 requires native E4M3
/// hardware (`mp_31`) and the muDNN route; there is no emulation fallback.
fn resolve_precision(
    requested: ModelPrecision,
    backend: &backend::RuntimeBackend,
    calibration: Option<&Path>,
) -> Result<ModelPrecision> {
    match requested {
        ModelPrecision::Auto | ModelPrecision::Bf16 => Ok(ModelPrecision::Bf16),
        ModelPrecision::Fp8 => {
            let family = backend.context().arch_family();
            if family != apxinf_musa::MusaArchFamily::Mp31 || apxinf_musa::FP8_ROUTE != "mudnn" {
                return Err(Error::Other(format!(
                    "GR00T FP8 requires an mp_31 build with the muDNN E4M3 route \
                     (device family {family}, FP8_ROUTE={})",
                    apxinf_musa::FP8_ROUTE
                )));
            }
            if calibration.is_none() {
                return Err(Error::Other(
                    "GR00T FP8 requires a calibration profile: pass a calibration path or \
                     place calibration.json in the model directory"
                        .into(),
                ));
            }
            Ok(ModelPrecision::Fp8)
        }
        other => Err(Error::Other(format!(
            "MUSA GR00T supports auto, bf16, and fp8, not {other:?}"
        ))),
    }
}

fn select_tactics(default: &Path, legacy: &Path, options: &LoadOptions) -> Option<PathBuf> {
    match options.tuning_path.as_ref() {
        Some(path) if path.is_file() => Some(path.clone()),
        Some(path) if options.autotune && !path.exists() => None,
        Some(path) => Some(path.clone()),
        None => default.is_file().then(|| default.to_path_buf())
            .or_else(|| legacy.is_file().then(|| legacy.to_path_buf())),
    }
}

fn install_tactics(runtime: &backend::RuntimeBackend, root: &Path, options: &LoadOptions) -> Result<()> {
    use backend::tuning::{configure_tuning, TuningDb, TuningMode, TuningPaths};

    let context = runtime.context();
    let defaults = TuningPaths::resolve_for_device(
        "configs/tuning", context.caps(), context.library_versions(),
    );
    let selected = select_tactics(&defaults.tactics, &root.join("tactics.json"), options);
    let database = selected.as_deref().map(TuningDb::from_json_file).transpose()?;
    let paths = options.tuning_path.clone().map(TuningPaths::from_tactics).unwrap_or(defaults);
    let mode = if options.autotune { TuningMode::AutoTune } else { TuningMode::Inference };
    configure_tuning(context, mode, database.as_slice(), Some(paths))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn validate_options(options: &LoadOptions) -> Result<()> {
    if options.synthetic.is_some()
        || options.config.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other(
            "GR00T does not support PI0.5 synthetic/config load options".into(),
        ));
    }
    match options.precision {
        ModelPrecision::Auto | ModelPrecision::Bf16 | ModelPrecision::Fp8 => Ok(()),
        precision => Err(Error::Other(format!(
            "MUSA GR00T supports auto, bf16, and fp8, not {precision:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_missing_tactics_are_only_created_by_autotune() {
        let directory = std::env::temp_dir().join(format!("gr00t-tactics-{}", std::process::id()));
        let path = directory.join("missing.json");
        let mut options = LoadOptions { tuning_path: Some(path.clone()), ..LoadOptions::default() };
        assert_eq!(select_tactics(&directory.join("default"), &directory.join("legacy"), &options), Some(path));
        options.autotune = true;
        assert_eq!(select_tactics(&directory.join("default"), &directory.join("legacy"), &options), None);
    }

    #[test]
    fn accepts_auto_and_bf16_precision() {
        for precision in [ModelPrecision::Auto, ModelPrecision::Bf16] {
            let options = LoadOptions {
                precision,
                ..LoadOptions::default()
            };
            validate_options(&options).unwrap();
        }
    }

    #[test]
    fn rejects_unimplemented_precisions() {
        let options = LoadOptions {
            precision: ModelPrecision::W8A8,
            ..LoadOptions::default()
        };
        assert!(validate_options(&options).is_err());
    }

    #[test]
    fn accepts_explicit_fp8_request() {
        let options = LoadOptions {
            precision: ModelPrecision::Fp8,
            ..LoadOptions::default()
        };
        validate_options(&options).unwrap();
    }
}
