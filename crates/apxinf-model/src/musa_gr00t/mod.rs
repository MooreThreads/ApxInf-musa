//! GR00T N1.7 on Moore Threads GPUs.
//!
//! This backend-native port owns its checkpoint interpretation and model
//! semantics. It intentionally does not import the CUDA `gr00t` module; both
//! ports are validated against the same reference fixtures instead.

mod action_weights;
mod assets;
mod backbone;
mod backend;
mod config;
mod geometry;
mod load;
mod math;
mod model;
mod model_runner;
mod weights;

pub(crate) use config::Gr00tConfig;
pub(crate) use load::load;
pub(crate) use model_runner::Gr00tModelRunner;
