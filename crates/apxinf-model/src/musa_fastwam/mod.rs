//! Fast-WAM on Moore Threads GPUs.
//!
//! This is a backend-native, eager-first port. It owns the Fast-WAM numerical
//! recipe and checkpoint mapping and does not import either CUDA GR00T or MUSA
//! GR00T model code. The only shared surface it uses is the safe, model-neutral
//! `apxinf-musa` operator API.

mod backend;
mod config;
mod geometry;
mod model;
mod runtime;
mod scheduler;
mod vae;
mod weights;

pub use config::{FastWamConfig, FastWamMode};
pub(crate) use backend::FastWamBackend;
pub(crate) use geometry::{sinusoidal_embedding_1d, FastWamGeometry, RopeTable};
pub(crate) use model::FastWamModel;
pub(crate) use scheduler::FlowMatchSchedule;
pub(crate) use vae::WanVae;
pub(crate) use weights::{FastWamWeights, WanVaeWeights};
pub(crate) use runtime::load as load_runtime;
