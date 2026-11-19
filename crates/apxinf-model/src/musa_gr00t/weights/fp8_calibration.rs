//! Static-FP8 calibration for the MUSA GR00T port.
//!
//! The shared schema is `apxinf.fp8-calibration.v1`, produced by the
//! model-neutral Python `CalibrationRunner` with `model_family="gr00t"`. This
//! module is the MUSA family's own reader: it validates the same document the
//! CUDA family does without importing that family, per the backend-native
//! port rule. The checkpoint identity is the same versioned value produced by
//! [`super::super::assets::checkpoint_identity`] and by
//! `Gr00tPolicy.checkpoint_identity` on the Python side.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

use super::super::assets::checkpoint_identity;

pub(super) const CALIBRATION_SCHEMA: &str = "apxinf.fp8-calibration.v1";
pub(super) const MODEL_FAMILY: &str = "gr00t";
const FP8_FORMAT: &str = "e4m3fn";
const CALIBRATION_STATISTIC: &str = "absmax";
pub(super) const CALIBRATION_SCALE_RULE: &str = "max(amax*margin/448,1e-8)";
pub(super) const E4M3_MAX: f32 = 448.0;

/// Per-tensor E4M3 weight scale mapping the largest magnitude onto the
/// largest representable E4M3 value. A zero tensor yields a unit scale so the
/// quantizer stays finite.
pub(super) fn e4m3_weight_scale(max_abs: f32) -> f32 {
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / E4M3_MAX
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDocument {
    schema: String,
    model: CalibrationModel,
    quantization: CalibrationQuantization,
    calibration_data: CalibrationData,
    seed_policy: CalibrationSeedPolicy,
    source_revision: String,
    device: BTreeMap<String, String>,
    plan: CalibrationPlan,
    observed_sites: Vec<String>,
    scales: BTreeMap<String, CalibrationScale>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationModel {
    family: String,
    checkpoint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationQuantization {
    format: String,
    statistic: String,
    scale_rule: String,
    margin: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationData {
    identity: String,
    kind: String,
    production: bool,
    sample_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSeedPolicy {
    algorithm: String,
    #[serde(rename = "base_seed")]
    _base_seed: u64,
    sample_sequence: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationPlan {
    sites: Vec<String>,
    consumers: BTreeMap<String, String>,
    statistics: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationScale {
    amax: f32,
    scale: f32,
}

/// Validated static-FP8 activation scales indexed by stable consumer ID.
#[derive(Debug)]
pub(crate) struct Gr00tFp8Calibration {
    activation_scales: BTreeMap<String, f32>,
}

impl Gr00tFp8Calibration {
    pub(crate) fn from_json_file(
        path: &Path,
        checkpoint_path: &Path,
        backbone_path: &Path,
        expected_consumers: &[String],
    ) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            Error::Other(format!(
                "read GR00T FP8 calibration {}: {error}",
                path.display()
            ))
        })?;
        let checkpoint = checkpoint_identity(checkpoint_path, backbone_path)?;
        Self::from_json_str(&raw, &checkpoint, expected_consumers)
    }

    fn from_json_str(raw: &str, checkpoint: &str, expected_consumers: &[String]) -> Result<Self> {
        let document: CalibrationDocument = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T FP8 calibration JSON: {error}")))?;
        if document.schema != CALIBRATION_SCHEMA {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration schema mismatch: expected {CALIBRATION_SCHEMA}, got {}",
                document.schema
            )));
        }
        if document.model.family != MODEL_FAMILY {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration model family mismatch: {}",
                document.model.family
            )));
        }
        if document.model.checkpoint != checkpoint {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration checkpoint identity mismatch: profile={}, runtime={checkpoint}; regenerate calibration for the current weights and processor resources",
                document.model.checkpoint
            )));
        }
        if document.quantization.format != FP8_FORMAT
            || document.quantization.statistic != CALIBRATION_STATISTIC
            || document.quantization.scale_rule != CALIBRATION_SCALE_RULE
            || !document.quantization.margin.is_finite()
            || document.quantization.margin < 1.0
        {
            return Err(Error::Other(
                "GR00T FP8 calibration quantization contract mismatch".into(),
            ));
        }
        if document.calibration_data.identity.is_empty()
            || document.calibration_data.sample_count == 0
            || document.source_revision.is_empty()
            || document.source_revision == "unknown"
            || document.seed_policy.algorithm.is_empty()
            || document.seed_policy.sample_sequence.is_empty()
            || document
                .device
                .get("requested")
                .map_or(true, String::is_empty)
            || document.device.get("host").map_or(true, String::is_empty)
        {
            return Err(Error::Other(
                "GR00T FP8 calibration manifest is incomplete".into(),
            ));
        }
        if document.calibration_data.kind != "representative"
            || !document.calibration_data.production
        {
            return Err(Error::Other(
                "GR00T FP8 calibration requires representative production data".into(),
            ));
        }

        let expected_consumers = expected_consumers.iter().cloned().collect::<BTreeSet<_>>();
        let expected_sites = expected_consumers
            .iter()
            .map(|consumer| format!("{consumer}.input"))
            .collect::<BTreeSet<_>>();
        let plan_sites = unique_strings("plan.sites", &document.plan.sites)?;
        let observed_sites = unique_strings("observed_sites", &document.observed_sites)?;
        let scale_sites = document.scales.keys().cloned().collect::<BTreeSet<_>>();
        let plan_consumers = document
            .plan
            .consumers
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if plan_sites != expected_sites
            || observed_sites != expected_sites
            || scale_sites != expected_sites
            || plan_consumers != expected_consumers
        {
            return Err(Error::Other(
                "GR00T FP8 calibration coverage is missing, unknown, or incompatible".into(),
            ));
        }
        for (consumer, site) in &document.plan.consumers {
            if site != &format!("{consumer}.input") {
                return Err(Error::Other(format!(
                    "GR00T FP8 consumer {consumer:?} maps to incompatible site {site:?}"
                )));
            }
        }
        if document.plan.statistics.len() != expected_sites.len()
            || document.plan.statistics.iter().any(|(site, statistic)| {
                !expected_sites.contains(site) || statistic != CALIBRATION_STATISTIC
            })
        {
            return Err(Error::Other(
                "GR00T FP8 calibration statistics do not match the execution plan".into(),
            ));
        }

        let mut activation_scales = BTreeMap::new();
        for consumer in expected_consumers {
            let site = &document.plan.consumers[&consumer];
            let entry = &document.scales[site];
            if !entry.amax.is_finite()
                || entry.amax < 0.0
                || !entry.scale.is_finite()
                || entry.scale <= 0.0
            {
                return Err(Error::Other(format!(
                    "GR00T FP8 calibration entry {site:?} is non-finite or non-positive"
                )));
            }
            let expected_scale = ((entry.amax as f64 * document.quantization.margin as f64)
                / E4M3_MAX as f64)
                .max(1.0e-8);
            if !expected_scale.is_finite()
                || expected_scale > f32::MAX as f64
                || (entry.scale as f64 - expected_scale).abs() > expected_scale * 1.0e-5
            {
                return Err(Error::Other(format!(
                    "GR00T FP8 calibration entry {site:?} has inconsistent amax/scale values"
                )));
            }
            activation_scales.insert(consumer, entry.scale);
        }
        Ok(Self { activation_scales })
    }

    pub(crate) fn scale(&self, consumer: &str) -> Result<f32> {
        self.activation_scales
            .get(consumer)
            .copied()
            .ok_or_else(|| {
                Error::Other(format!(
                    "GR00T FP8 calibration is missing consumer {consumer:?}"
                ))
            })
    }
}

fn unique_strings(label: &str, values: &[String]) -> Result<BTreeSet<String>> {
    let set = values.iter().cloned().collect::<BTreeSet<_>>();
    if set.len() != values.len() {
        return Err(Error::Other(format!(
            "GR00T FP8 calibration {label} contains duplicate values"
        )));
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> String {
        "sha256:0000000000000000000000000000000000000000000000000000000000000000".into()
    }

    fn document(consumers: &[&str], amax: f32, scale: f32) -> String {
        let sites = consumers
            .iter()
            .map(|consumer| format!("{consumer}.input"))
            .collect::<Vec<_>>();
        let consumers_json = consumers
            .iter()
            .map(|consumer| format!("\"{consumer}\":\"{consumer}.input\""))
            .collect::<Vec<_>>()
            .join(",");
        let statistics = sites
            .iter()
            .map(|site| format!("\"{site}\":\"absmax\""))
            .collect::<Vec<_>>()
            .join(",");
        let scales = sites
            .iter()
            .map(|site| format!("\"{site}\":{{\"amax\":{amax},\"scale\":{scale}}}"))
            .collect::<Vec<_>>()
            .join(",");
        let sites_json = sites
            .iter()
            .map(|site| format!("\"{site}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{CALIBRATION_SCHEMA}\",\"model\":{{\"family\":\"{MODEL_FAMILY}\",\"checkpoint\":\"{id}\"}},\
             \"quantization\":{{\"format\":\"{FP8_FORMAT}\",\"statistic\":\"{CALIBRATION_STATISTIC}\",\"scale_rule\":\"{CALIBRATION_SCALE_RULE}\",\"margin\":1.1}},\
             \"calibration_data\":{{\"identity\":\"fixture\",\"kind\":\"representative\",\"production\":true,\"sample_count\":1}},\
             \"seed_policy\":{{\"algorithm\":\"numpy-pcg64\",\"base_seed\":0,\"sample_sequence\":\"seed,index\"}},\
             \"source_revision\":\"deadbeef\",\"device\":{{\"requested\":\"musa:0\",\"host\":\"test\"}},\
             \"plan\":{{\"sites\":[{sites_json}],\"consumers\":{{{consumers_json}}},\"statistics\":{{{statistics}}}}},\
             \"observed_sites\":[{sites_json}],\"scales\":{{{scales}}}}}",
            id = identity()
        )
    }

    #[test]
    fn accepts_a_complete_profile() {
        let consumers = ["backbone.text.layers.0.query"];
        let amax = 2.0f32;
        let scale = amax * 1.1 / E4M3_MAX;
        let parsed = Gr00tFp8Calibration::from_json_str(
            &document(&consumers, amax, scale),
            &identity(),
            &[consumers[0].to_string()],
        )
        .unwrap();
        assert_eq!(parsed.scale(consumers[0]).unwrap(), scale);
    }

    #[test]
    fn rejects_missing_consumer_coverage() {
        let consumers = ["backbone.text.layers.0.query"];
        let amax = 2.0f32;
        let scale = amax * 1.1 / E4M3_MAX;
        let error = Gr00tFp8Calibration::from_json_str(
            &document(&consumers, amax, scale),
            &identity(),
            &[
                consumers[0].to_string(),
                "backbone.text.layers.0.key".into(),
            ],
        )
        .unwrap_err();
        assert!(error.to_string().contains("coverage"), "{error}");
    }

    #[test]
    fn rejects_other_model_family() {
        let consumers = ["backbone.text.layers.0.query"];
        let raw = document(&consumers, 2.0, 2.0 * 1.1 / E4M3_MAX)
            .replace("\"family\":\"gr00t\"", "\"family\":\"pi05\"");
        let error = Gr00tFp8Calibration::from_json_str(&raw, &identity(), &[consumers[0].into()])
            .unwrap_err();
        assert!(error.to_string().contains("model family"), "{error}");
    }

    #[test]
    fn rejects_inconsistent_amax_and_scale() {
        let consumers = ["backbone.text.layers.0.query"];
        let error = Gr00tFp8Calibration::from_json_str(
            &document(&consumers, 2.0, 1.0),
            &identity(),
            &[consumers[0].into()],
        )
        .unwrap_err();
        assert!(error.to_string().contains("inconsistent"), "{error}");
    }

    #[test]
    fn rejects_non_production_data() {
        let consumers = ["backbone.text.layers.0.query"];
        let raw = document(&consumers, 2.0, 2.0 * 1.1 / E4M3_MAX)
            .replace("\"production\":true", "\"production\":false");
        let error = Gr00tFp8Calibration::from_json_str(&raw, &identity(), &[consumers[0].into()])
            .unwrap_err();
        assert!(error.to_string().contains("representative"), "{error}");
    }

    #[test]
    fn zero_weight_scale_is_unit() {
        assert_eq!(e4m3_weight_scale(0.0), 1.0);
        assert_eq!(e4m3_weight_scale(448.0), 1.0);
        assert_eq!(e4m3_weight_scale(224.0), 0.5);
    }
}
