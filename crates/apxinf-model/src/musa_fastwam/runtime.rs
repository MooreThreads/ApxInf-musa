//! Public VLA seam for the eager Fast-WAM MUSA implementation.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Device, DType, Error, Graph, Result, Tensor};
use apxinf_musa::ops;
use apxinf_musa::transfers::to_cpu;

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, ExecutionPolicy, InferenceOutputs, InferenceSpec, InitialLatent, PreparedInference,
    ExecutionMode, VlaContract, VlaRequest, VlaRuntime, VisionObservation,
};

use super::{FastWamConfig, FastWamMode, FastWamModel, FastWamWeights, WanVae, WanVaeWeights};

pub(crate) struct FastWamRuntime {
    config: FastWamConfig,
    model: Rc<FastWamModel>,
    vae: Rc<WanVae>,
    execution_policy: Cell<ExecutionPolicy>,
    prepared: RefCell<Option<Rc<FastWamPreparedPlan>>>,
    graph_setup_failed: Cell<bool>,
}

/// Device-resident, fixed-profile inputs. The tensors are intentionally kept
/// alive for the lifetime of a captured graph because the graph records their
/// addresses, not their Rust handles.
struct FastWamPreparedInputs {
    image: Tensor,
    context: Tensor,
    proprio: Tensor,
    action_noise: Tensor,
    video_noise: Option<Tensor>,
}

struct FastWamPreparedPlan {
    spec: InferenceSpec,
    model: Rc<FastWamModel>,
    vae: Rc<WanVae>,
    inputs: FastWamPreparedInputs,
    vae_session: apxinf_musa::ExecutionSession,
    model_session: apxinf_musa::ExecutionSession,
    _first_frame: Tensor,
    vae_graph: Option<apxinf_musa::CapturedGraph>,
    vae_graph_output: Option<Tensor>,
    graph: Option<apxinf_musa::CapturedGraph>,
    graph_output: Option<Tensor>,
    fallback_reason: Option<String>,
}

struct FastWamPreparedInference {
    plan: Rc<FastWamPreparedPlan>,
}

pub(crate) fn load(path: &Path, device: Device, options: &LoadOptions) -> Result<LoadedModel> {
    let Device::Musa(device_id) = device else {
        return Err(Error::Other(format!("Fast-WAM requires MUSA, not {device}")));
    };
    if !matches!(options.precision, ModelPrecision::Auto | ModelPrecision::Bf16) {
        return Err(Error::Other("MUSA Fast-WAM currently supports BF16 only".into()));
    }
    if options.synthetic.is_some() || options.calibration_path.is_some() {
        return Err(Error::Other(
            "MUSA Fast-WAM does not support synthetic or calibration loading".into(),
        ));
    }
    let mode = options
        .model_variant
        .as_deref()
        .unwrap_or("fast")
        .parse::<FastWamMode>()
        .map_err(|_| Error::Other("Fast-WAM model_variant must be fast, idm, or joint".into()))?;
    let config = FastWamConfig::libero_2cam224(mode);
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let model_path = options
        .assets
        .get("model")
        .cloned()
        .unwrap_or_else(|| root.join("model"));
    let vae_path = options
        .assets
        .get("vae")
        .cloned()
        .unwrap_or_else(|| root.join("vae"));
    let model_weights = FastWamWeights::from_safetensors(&config, &model_path)?;
    let vae_weights = WanVaeWeights::from_safetensors(&vae_path)?;
    let model = FastWamModel::from_host(device_id, config.clone(), model_weights)?;
    let vae = WanVae::from_host(model.backend().clone(), &config, vae_weights)?;
    let execution_policy = options.execution_policy.unwrap_or(ExecutionPolicy::Eager);
    let runtime = FastWamRuntime {
        config,
        model: Rc::new(model),
        vae: Rc::new(vae),
        execution_policy: Cell::new(execution_policy),
        prepared: RefCell::new(None),
        graph_setup_failed: Cell::new(false),
    };
    Ok(LoadedModel::Vla(Box::new(runtime)))
}

impl FastWamRuntime {
    fn build_inputs(&self, request: &VlaRequest<'_>) -> Result<FastWamPreparedInputs> {
        let image = match &request.observation.vision {
            VisionObservation::Patches(image) => image,
            VisionObservation::RgbU8 { .. } => {
                return Err(Error::Other(
                    "Fast-WAM runtime expects preprocessed NCHW image tensor".into(),
                ))
            }
        };
        let state = request
            .observation
            .state
            .as_ref()
            .ok_or_else(|| Error::Other("Fast-WAM requires context in observation.state".into()))?;
        let proprio = request
            .observation
            .action_mask
            .as_ref()
            .ok_or_else(|| Error::Other("Fast-WAM requires proprio in observation.action_mask".into()))?;
        let InitialLatent::Provided(action_noise) = request.initial_latent else {
            return Err(Error::Other(
                "Fast-WAM requires caller-provided deterministic action noise".into(),
            ));
        };
        if let Some(mask) = request.metadata.attention_mask {
            if mask.len() != self.config.context_len || mask.iter().any(|&v| v == 0) {
                return Err(Error::Other(
                    "MUSA Fast-WAM first release requires an all-valid 128-token context mask".into(),
                ));
            }
        }
        let image = self.to_device_bf16(image, "input_image")?;
        let context = self.to_device_bf16(state, "context")?;
        let proprio = self.to_device_bf16(proprio, "proprio")?;
        let action_noise = self.to_device_bf16(action_noise, "action_noise")?;
        let video_noise = request
            .metadata
            .auxiliary_latent
            .map(|noise| self.to_device_bf16(noise, "video_noise"))
            .transpose()?;
        if self.config.mode != FastWamMode::Fast && video_noise.is_none() {
            return Err(Error::Other(
                "Fast-WAM IDM/Joint requires deterministic video_noise".into(),
            ));
        }
        Ok(FastWamPreparedInputs {
            image,
            context,
            proprio,
            action_noise,
            video_noise,
        })
    }

    fn to_device_bf16(&self, tensor: &Tensor, name: &str) -> Result<Tensor> {
        let tensor = to_bf16(tensor, name)?;
        if tensor.device() == Device::Cpu {
            self.model.backend().to_device(&tensor)
        } else {
            Ok(tensor)
        }
    }

    fn run_device(&self, inputs: &FastWamPreparedInputs) -> Result<Tensor> {
        Self::run_device_with(&self.model, &self.vae, inputs)
    }

    fn run_device_with(
        model: &FastWamModel,
        vae: &WanVae,
        inputs: &FastWamPreparedInputs,
    ) -> Result<Tensor> {
        let first_frame = vae.encode_first_frame(&inputs.image)?;
        let output = model.infer_eager_from_latents(
            &inputs.context,
            &inputs.proprio,
            &first_frame,
            &inputs.action_noise,
            inputs.video_noise.as_ref(),
        )?;
        Ok(output.action)
    }

    // VAE and DiT are separate prepared sessions.  Keep the historical
    // variable as the fallback, but allow the much larger IDM/Joint DiT arena
    // without allocating the same capacity a second time for the VAE.
    fn vae_workspace_bytes() -> usize {
        std::env::var("APXINF_FASTWAM_VAE_WORKSPACE_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .or_else(|| {
                std::env::var("APXINF_FASTWAM_GRAPH_WORKSPACE_BYTES")
                    .ok()
                    .and_then(|value| value.parse().ok())
            })
            .unwrap_or(4usize << 30)
    }

    fn dit_workspace_bytes() -> usize {
        std::env::var("APXINF_FASTWAM_DIT_WORKSPACE_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .or_else(|| {
                std::env::var("APXINF_FASTWAM_GRAPH_WORKSPACE_BYTES")
                    .ok()
                    .and_then(|value| value.parse().ok())
            })
            .unwrap_or(8usize << 30)
    }

    fn build_prepared_plan(
        &self,
        request: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Rc<FastWamPreparedPlan>> {
        let inputs = self.build_inputs(request)?;
        let vae_session = apxinf_musa::ExecutionSession::with_bump_capacity(
            Self::vae_workspace_bytes(),
            self.model.backend().device_id(),
        )?;
        let model_session = apxinf_musa::ExecutionSession::with_bump_capacity(
            Self::dit_workspace_bytes(),
            self.model.backend().device_id(),
        )?;

        // Prime the VAE in its own prepared arena.  The VAE and DiT sessions
        // are separate because the VAE latent must remain at a stable address
        // while the DiT graph replays; this is a phase split, not a separate
        // "prepared arena" versus "graph arena" abstraction.
        let first_frame = ops::prepare_with_session(&vae_session, || {
            self.vae.encode_first_frame(&inputs.image)
        })
        .map_err(|error| Error::Other(format!("Fast-WAM VAE preflight failed: {error}")))?;
        self.model.backend().synchronize()?;

        if policy == ExecutionPolicy::Eager {
            return Ok(Rc::new(FastWamPreparedPlan {
                spec: request.observation.inference_spec(),
                model: Rc::clone(&self.model),
                vae: Rc::clone(&self.vae),
                inputs,
                vae_session,
                model_session,
                _first_frame: first_frame,
                vae_graph: None,
                vae_graph_output: None,
                graph: None,
                graph_output: None,
                fallback_reason: None,
            }));
        }

        let context = self.model.backend().context_arc();
        // Capture VAE immediately after its own preflight.  Keeping this
        // adjacent is important for muDNN: a later, unrelated DiT traversal
        // can touch stream-scoped vendor state before the VAE capture starts.
        let vae_captured = apxinf_musa::capture_isolated(Arc::clone(&context), || {
            ops::with_session(&vae_session, || self.vae.encode_first_frame(&inputs.image))
        });
        let (vae_graph, captured_first_frame, vae_graph_reason) = match vae_captured {
            Ok((graph, output)) => (Some(graph), Some(output), None),
            Err(error) if policy == ExecutionPolicy::RequireGraph => {
                return Err(Error::Other(format!(
                    "Fast-WAM VAE MUSA graph capture is required but failed: {error}"
                )));
            }
            Err(error) => {
                eprintln!(
                    "[apxinf] Fast-WAM VAE graph capture unavailable, using prepared eager: {error}"
                );
                (None, None, Some(error.to_string()))
            }
        };

        let dit_first_frame = captured_first_frame.as_ref().unwrap_or(&first_frame);

        // Prime DiT provider descriptors/tactics and lay out every temporary
        // before capture. Only this model traversal is recorded; request
        // latency still includes the prepared VAE traversal above.
        let eager_output = ops::prepare_with_session(&model_session, || {
            self.model.infer_eager_from_latents(
                &inputs.context,
                &inputs.proprio,
                dit_first_frame,
                &inputs.action_noise,
                inputs.video_noise.as_ref(),
            )
        })
        .map(|output| output.action)
        .map_err(|error| Error::Other(format!("Fast-WAM DiT graph preflight failed: {error}")))?;
        self.model.backend().synchronize()?;
        drop(eager_output);

        let mut captured_output = None;
        let captured = apxinf_musa::capture(context, || {
            captured_output = Some(ops::with_session(&model_session, || {
                self.model
                    .infer_eager_from_latents(
                        &inputs.context,
                        &inputs.proprio,
                        dit_first_frame,
                        &inputs.action_noise,
                        inputs.video_noise.as_ref(),
                    )
                    .map(|output| output.action)
            })?);
            Ok(())
        });
        match captured {
            Ok(graph) => {
                eprintln!("[apxinf] Fast-WAM MUSA graph captured");
                Ok(Rc::new(FastWamPreparedPlan {
                    spec: request.observation.inference_spec(),
                    model: Rc::clone(&self.model),
                    vae: Rc::clone(&self.vae),
                    inputs,
                    vae_session,
                    model_session,
                    _first_frame: first_frame,
                    vae_graph,
                    vae_graph_output: captured_first_frame,
                    graph: Some(graph),
                    graph_output: Some(captured_output.expect("capture produced Fast-WAM output")),
                    fallback_reason: vae_graph_reason,
                }))
            }
            Err(error) if policy == ExecutionPolicy::PreferGraph => {
                let reason = error.to_string();
                eprintln!(
                    "[apxinf] Fast-WAM MUSA graph capture unavailable, using prepared eager: {reason}"
                );
                Ok(Rc::new(FastWamPreparedPlan {
                    spec: request.observation.inference_spec(),
                    model: Rc::clone(&self.model),
                    vae: Rc::clone(&self.vae),
                    inputs,
                    vae_session,
                    model_session,
                    _first_frame: first_frame,
                    vae_graph: None,
                    vae_graph_output: None,
                    graph: None,
                    graph_output: None,
                    fallback_reason: Some(reason),
                }))
            }
            Err(error) => Err(Error::Other(format!(
                "Fast-WAM MUSA graph capture is required but failed: {error}"
            ))),
        }
    }

    fn run_prepared_plan(&self, plan: &FastWamPreparedPlan) -> Result<Action> {
        plan.run()
    }

    fn prepared_for_request(&self, request: &VlaRequest<'_>) -> Result<Rc<FastWamPreparedPlan>> {
        let spec = request.observation.inference_spec();
        if let Some(plan) = self.prepared.borrow().as_ref() {
            if plan.spec == spec {
                return Ok(Rc::clone(plan));
            }
        }
        let plan = self.build_prepared_plan(request, self.execution_policy.get())?;
        *self.prepared.borrow_mut() = Some(Rc::clone(&plan));
        Ok(plan)
    }
}

impl FastWamPreparedPlan {
    fn run(&self) -> Result<Action> {
        // Re-run VAE into the same deterministic arena slices whose final
        // latent address was captured by the DiT graph.
        let first_frame = if let Some(graph) = &self.vae_graph {
            graph.replay()?;
            self.vae_graph_output
                .as_ref()
                .expect("captured Fast-WAM VAE graph has an output")
                .clone()
        } else {
            ops::with_session(&self.vae_session, || {
                self.vae.encode_first_frame(&self.inputs.image)
            })?
        };
        let output = if let Some(graph) = &self.graph {
            graph.replay()?;
            self.graph_output
                .as_ref()
                .expect("captured Fast-WAM graph has an output")
                .clone()
        } else {
            let output = ops::with_session(&self.model_session, || {
                self.model
                    .infer_eager_from_latents(
                        &self.inputs.context,
                        &self.inputs.proprio,
                        &first_frame,
                        &self.inputs.action_noise,
                        self.inputs.video_noise.as_ref(),
                    )
                    .map(|output| output.action)
            })?;
            output
        };
        Ok(Action::new(output))
    }

}

impl VlaRuntime for FastWamRuntime {
    fn model_variant(&self) -> Option<&'static str> {
        Some(match self.config.mode {
            FastWamMode::Fast => "fast",
            FastWamMode::Idm => "idm",
            FastWamMode::Joint => "joint",
        })
    }

    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [self.config.action_horizon, self.config.action_dim],
            patch_shape: [0, 0],
            max_token_len: self.config.context_len,
            num_views: self.config.num_views,
            image_size: self.config.camera_width,
            patch_size: 1,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        Ok(self.infer_with_outputs(request)?.action)
    }

    fn infer_action_only(&self, request: &VlaRequest<'_>) -> Result<Action> {
        match self.execution_policy.get() {
            ExecutionPolicy::Eager => {
                let inputs = self.build_inputs(request)?;
                Ok(Action::new(self.run_device(&inputs)?))
            }
            ExecutionPolicy::PreferGraph | ExecutionPolicy::RequireGraph => {
                if self.execution_policy.get() == ExecutionPolicy::PreferGraph
                    && self.graph_setup_failed.get()
                {
                    let inputs = self.build_inputs(request)?;
                    return Ok(Action::new(self.run_device(&inputs)?));
                }
                match self.prepared_for_request(request) {
                    Ok(plan) => self.run_prepared_plan(&plan),
                    Err(error) if self.execution_policy.get() == ExecutionPolicy::PreferGraph => {
                        // A graph attempt may fail before a prepared traversal
                        // exists (for example, when the fixed arena is too
                        // small). PreferGraph promises an eager fallback even
                        // for that preflight failure.
                        eprintln!(
                            "[apxinf] Fast-WAM prepared graph setup failed, using eager: {error}"
                        );
                        self.graph_setup_failed.set(true);
                        let inputs = self.build_inputs(request)?;
                        Ok(Action::new(self.run_device(&inputs)?))
                    }
                    Err(error) => Err(error),
                }
            }
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.model.backend().synchronize()
    }

    fn infer_with_outputs(&self, request: &VlaRequest<'_>) -> Result<InferenceOutputs> {
        let inputs = self.build_inputs(request)?;
        let first_frame = self.vae.encode_first_frame(&inputs.image)?;
        let output = self.model.infer_eager_from_latents(
            &inputs.context,
            &inputs.proprio,
            &first_frame,
            &inputs.action_noise,
            inputs.video_noise.as_ref(),
        )?;
        let decoded_video = output
            .video_latents
            .as_ref()
            .map(|latents| self.vae.decode(latents))
            .transpose()?;
        Ok(InferenceOutputs {
            action: Action::new(to_cpu(&output.action)?),
            video_latents: output.video_latents.as_ref().map(to_cpu).transpose()?,
            decoded_video: decoded_video.as_ref().map(to_cpu).transpose()?,
        })
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        if spec.image_layout.is_some() {
            return Err(Error::Other(
                "Fast-WAM prepared inference accepts preprocessed NCHW input only".into(),
            ));
        }
        Err(Error::Other(
            "Fast-WAM preparation needs a real fixed input; use prepare_for".into(),
        ))
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        _policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        Err(Error::Other(
            "Fast-WAM preparation needs a real fixed input; use prepare_for".into(),
        ))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        let plan = self.build_prepared_plan(sample, policy)?;
        Ok(Box::new(FastWamPreparedInference { plan }))
    }

    fn clear_prepared(&self) -> Result<()> {
        self.model.backend().synchronize()?;
        drop(self.prepared.borrow_mut().take());
        self.graph_setup_failed.set(false);
        Ok(())
    }

    fn execution_mode(&self) -> &'static str {
        match self.prepared.borrow().as_ref() {
            Some(plan) if plan.graph.is_some() => "musa-graph",
            Some(_) => "no_graph",
            None => "no_graph",
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer(request)?.tensor().to_f32_vec()
    }
}

impl PreparedInference for FastWamPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.plan.spec
    }

    fn status(&self) -> crate::vla::PreparationStatus {
        match self.plan.graph.is_some() {
            true => crate::vla::PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                // A graph plan may still contain the prepared-eager VAE when
                // muDNN rejects VAE capture. Preserve that reason for callers
                // inspecting preparation status; the DiT graph remains valid.
                fallback_reason: self.plan.fallback_reason.clone(),
            },
            false => crate::vla::PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: self.plan.fallback_reason.clone(),
            },
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec().matches(request.observation) {
            return Err(Error::Other(format!(
                "Fast-WAM prepared profile expects {:?}, got {:?}",
                self.spec(),
                request.observation.inference_spec()
            )));
        }
        self.plan.run()
    }
}

fn to_bf16(tensor: &Tensor, name: &str) -> Result<Tensor> {
    match tensor.dtype() {
        DType::BF16 => Ok(tensor.clone()),
        DType::F32 if tensor.device() == Device::Cpu => Tensor::from_bf16(
            tensor.shape().dims().to_vec(),
            &tensor
                .to_f32_vec()?
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        dtype => Err(Error::Other(format!(
            "Fast-WAM {name} must be CPU f32 or bf16, got {dtype} on {}",
            tensor.device()
        ))),
    }
}
