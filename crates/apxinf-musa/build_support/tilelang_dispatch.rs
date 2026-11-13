//! Generate typed static dispatch tables from the TileLang inventory.
//!
//! Operator types declare their stages and ABIs here. Rendering declarations,
//! collecting objects and constructing table rows use the same code for every
//! type; precision-specific launch and preparation stay in the runtime module.

use std::path::{Path, PathBuf};

use serde_json::Value;

struct Stage {
    role: &'static str,
    parameters: &'static [(&'static str, &'static str)],
    optional: bool,
}

struct OperatorType {
    name: &'static str,
    table: &'static str,
    row_type: &'static str,
    stages: &'static [Stage],
}

const PTR: &str = "*mut core::ffi::c_void";

const PART: Stage = Stage {
    role: "part",
    parameters: &[("a", PTR), ("b", PTR), ("out", PTR), ("stream", PTR)],
    optional: false,
};

const OPERATOR_TYPES: &[OperatorType] = &[
    OperatorType {
        name: "gemm-splitk-int8",
        table: "TILELANG_GEMMS",
        row_type: "TilelangGemm",
        stages: &[
            PART,
            Stage {
                role: "reduce",
                parameters: &[
                    ("partials", PTR),
                    ("a_scale", PTR),
                    ("b_scale", PTR),
                    ("out", PTR),
                    ("stream", PTR),
                ],
                optional: false,
            },
        ],
    },
    OperatorType {
        name: "gemm-bf16",
        table: "TILELANG_BF16_GEMMS",
        row_type: "TilelangBf16Gemm",
        stages: &[
            PART,
            Stage {
                role: "pack",
                parameters: &[("input", PTR), ("output", PTR), ("stream", PTR)],
                optional: false,
            },
            Stage {
                role: "reduce",
                parameters: &[("input", PTR), ("output", PTR), ("stream", PTR)],
                optional: true,
            },
        ],
    },
];

pub struct Dispatch {
    pub source: String,
    pub objects: Vec<PathBuf>,
}

/// The same renderer handles populated inventories and feature-disabled builds.
pub fn generate(artifacts: &[Value], directory: &Path) -> Result<Dispatch, String> {
    let mut declarations = String::new();
    let mut objects = Vec::new();
    let mut rows = vec![Vec::new(); OPERATOR_TYPES.len()];

    for artifact in artifacts {
        let kind = artifact["type"].as_str().ok_or("missing operator type")?;
        let index = OPERATOR_TYPES
            .iter()
            .position(|spec| spec.name == kind)
            .ok_or_else(|| format!("unsupported TileLang operator type: {kind}"))?;
        let spec = &OPERATOR_TYPES[index];
        let entries = artifact["entries"]
            .as_object()
            .ok_or("missing operator entries")?;
        for role in entries.keys() {
            if !spec.stages.iter().any(|stage| stage.role == role) {
                return Err(format!("{kind}: unexpected stage {role}"));
            }
        }
        let shape = artifact["shape"].as_array().ok_or("missing GEMM shape")?;
        if shape.len() != 3 || shape.iter().any(|dim| dim.as_u64().is_none()) {
            return Err(format!("{kind}: expected three unsigned GEMM dimensions"));
        }
        let status = match artifact["status"].as_str().unwrap_or("off") {
            "enabled" => "TilelangStatus::Enabled",
            "experimental" => "TilelangStatus::Experimental",
            _ => "TilelangStatus::Off",
        };
        let mut row = format!(
            "    {} {{ m: {}, n: {}, k: {}, status: {status}, scratch: {}, ",
            spec.row_type,
            shape[0],
            shape[1],
            shape[2],
            artifact["scratch_bytes"].as_u64().unwrap_or(0),
        );
        for stage in spec.stages {
            let role = stage.role;
            let entry = entries.get(role).and_then(Value::as_str);
            if entry.is_none() && !stage.optional {
                return Err(format!("{kind}: missing required stage {role}"));
            }
            if let Some(entry) = entry {
                let file = artifact["files"][role]
                    .as_str()
                    .ok_or_else(|| format!("{kind}: missing object for {role}"))?;
                objects.push(directory.join(file));
                let parameters = stage
                    .parameters
                    .iter()
                    .map(|(name, ty)| format!("{name}: {ty}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                declarations.push_str(&format!(
                    "    pub(super) fn {entry}({parameters}) -> i32;\n\
                     pub(super) fn {entry}_init() -> i32;\n"
                ));
            }
            let (function, init) = match (entry, stage.optional) {
                (Some(entry), false) => (entry.to_string(), format!("{entry}_init")),
                (Some(entry), true) => (format!("Some({entry})"), format!("Some({entry}_init)")),
                (None, _) => ("None".into(), "None".into()),
            };
            row.push_str(&format!(
                "{role}: {function}, {role}_init: {init}, \
                 {role}_ready: AtomicBool::new(false), "
            ));
        }
        row.push_str("},\n");
        rows[index].push(row);
    }

    let mut source = String::from("// Generated from the TileLang inventory. Do not edit.\n");
    if !declarations.is_empty() {
        source.push_str(&format!("extern \"C\" {{\n{declarations}}}\n\n"));
    }
    for (spec, rows) in OPERATOR_TYPES.iter().zip(rows) {
        source.push_str(&format!(
            "pub(super) static {}: [{}; {}] = [\n{}];\n",
            spec.table,
            spec.row_type,
            rows.len(),
            rows.concat(),
        ));
    }
    Ok(Dispatch { source, objects })
}
