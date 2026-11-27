//! Eager Fast-WAM DiT/MoT composition on safe MUSA operations.
//!
//! The block split follows upstream `mot.py` at revision
//! `7faa71108368fbb3b6885649f112af607427a2d4`: build expert Q/K/V after
//! token-wise AdaLN and RoPE, perform one masked mixed attention, then apply
//! expert-local output projection, cross attention and GELU-tanh FFN.

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_musa::ops::{self, BinaryOp, ConvolutionArgs, UnaryOp};
use std::cell::RefCell;

use super::weights::{
    AttentionWeights, DiTBlockWeights, EmbeddingWeights, ExpertWeights, LinearWeights,
};
use super::{
    sinusoidal_embedding_1d, FastWamBackend, FastWamConfig, FastWamGeometry, FastWamWeights,
    FlowMatchSchedule, RopeTable,
};

pub(crate) struct ExpertPrepared {
    pub(crate) tokens: Tensor,
    pub(crate) time: Tensor,
    pub(crate) modulation: Tensor,
    pub(crate) context: Tensor,
    pub(crate) context_mask: Tensor,
}

struct ExpertAttentionIo {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    residual: Tensor,
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

pub(crate) struct VideoKvCache {
    layers: Vec<(Tensor, Tensor)>,
    video_tokens: usize,
}

pub(crate) struct FastWamEagerOutput {
    pub(crate) action: Tensor,
    pub(crate) video_latents: Option<Tensor>,
}

pub(crate) struct FastWamModel {
    config: FastWamConfig,
    geometry: FastWamGeometry,
    backend: FastWamBackend,
    weights: FastWamWeights,
    video_rope_cos: Tensor,
    video_rope_sin: Tensor,
    action_rope_cos: Tensor,
    action_rope_sin: Tensor,
    video_attention_mask: Tensor,
    mot_attention_mask: Tensor,
    video_ones: Tensor,
    action_ones: Tensor,
    video_context_mask: Tensor,
    action_context_mask: Tensor,
    preserve_future_mask: Tensor,
    video_time_sinusoids: Vec<Tensor>,
    action_time_sinusoids: Vec<Tensor>,
    video_zero_sinusoid: Tensor,
    video_deltas: Vec<Tensor>,
    action_deltas: Vec<Tensor>,
    first_frame_zero_rows: RefCell<Option<Tensor>>,
    // Fixed-address state buffers.  Denoise-step temporaries are rewound after
    // each step; these ping-pong buffers are the only state that crosses the
    // rewind boundary.
    video_state_a: Tensor,
    video_state_b: Tensor,
    video_state_c: Tensor,
    action_state_a: Tensor,
    action_state_b: Tensor,
}

impl FastWamModel {
    pub(crate) fn from_host(
        device_id: usize,
        config: FastWamConfig,
        weights: FastWamWeights,
    ) -> Result<Self> {
        config.validate()?;
        let geometry = FastWamGeometry::new(&config)?;
        let backend = FastWamBackend::new(device_id)?;
        let weights = weights.into_device(&backend)?;
        let video_rope = geometry.video_rope(config.mode)?;
        let action_rope = geometry.action_rope()?;
        let video_rope_cos = upload_rope(&backend, &video_rope, true)?;
        let video_rope_sin = upload_rope(&backend, &video_rope, false)?;
        let action_rope_cos = upload_rope(&backend, &action_rope, true)?;
        let action_rope_sin = upload_rope(&backend, &action_rope, false)?;
        let video_tokens = geometry.inference_video_tokens(config.mode);
        let video_mask = geometry.video_attention_mask(video_tokens)?;
        let mot_mask = geometry.mot_attention_mask(config.mode)?;
        let video_attention_mask = backend.to_device(&Tensor::from_i8(
            vec![video_mask.rows(), video_mask.cols()],
            video_mask.values(),
        )?)?;
        let mot_attention_mask = backend.to_device(&Tensor::from_i8(
            vec![mot_mask.rows(), mot_mask.cols()],
            mot_mask.values(),
        )?)?;
        let video_ones = upload_ones(&backend, config.video_hidden_dim)?;
        let action_ones = upload_ones(&backend, config.action_hidden_dim)?;
        let video_schedule = FlowMatchSchedule::new(
            config.num_inference_steps,
            config.num_train_timesteps,
            config.video_shift,
        )?;
        let action_schedule = FlowMatchSchedule::new(
            config.num_inference_steps,
            config.num_train_timesteps,
            config.action_shift,
        )?;
        let context_tokens = config.context_len + 1;
        let video_context_mask = upload_true_mask(&backend, video_tokens, context_tokens)?;
        let action_context_mask = upload_true_mask(
            &backend,
            config.action_horizon,
            context_tokens,
        )?;
        let mut preserve_future = vec![half::bf16::ONE; config.video_latent_shape().iter().product()];
        let frame_elements = config.latent_height * config.latent_width;
        for channel in 0..config.latent_channels {
            let start = channel * config.latent_temporal * frame_elements;
            preserve_future[start..start + frame_elements].fill(half::bf16::ZERO);
        }
        let preserve_future_mask = backend.to_device(&Tensor::from_bf16(
            config.video_latent_shape().to_vec(),
            &preserve_future,
        )?)?;
        let video_time_sinusoids = video_schedule
            .timesteps()
            .iter()
            .map(|&timestep| {
                upload_sinusoid(
                    &backend,
                    config.freq_dim,
                    &geometry.video_token_timesteps(config.mode, timestep),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let action_time_sinusoids = action_schedule
            .timesteps()
            .iter()
            .map(|&timestep| upload_sinusoid(&backend, config.freq_dim, &[timestep]))
            .collect::<Result<Vec<_>>>()?;
        let video_zero_sinusoid = upload_sinusoid(
            &backend,
            config.freq_dim,
            &geometry.video_token_timesteps(config.mode, 0.0),
        )?;
        let video_deltas = upload_scalars(&backend, video_schedule.deltas())?;
        let action_deltas = upload_scalars(&backend, action_schedule.deltas())?;
        let video_state_a = backend.to_device(&Tensor::zeros(
            config.video_latent_shape().to_vec(),
            DType::BF16,
        ))?;
        let video_state_b = backend.to_device(&Tensor::zeros(
            config.video_latent_shape().to_vec(),
            DType::BF16,
        ))?;
        let video_state_c = backend.to_device(&Tensor::zeros(
            config.video_latent_shape().to_vec(),
            DType::BF16,
        ))?;
        let action_shape = vec![config.action_horizon, config.action_dim];
        let action_state_a = backend.to_device(&Tensor::zeros(action_shape.clone(), DType::BF16))?;
        let action_state_b = backend.to_device(&Tensor::zeros(action_shape, DType::BF16))?;
        Ok(Self {
            config,
            geometry,
            backend,
            weights,
            video_rope_cos,
            video_rope_sin,
            action_rope_cos,
            action_rope_sin,
            video_attention_mask,
            mot_attention_mask,
            video_ones,
            action_ones,
            video_context_mask,
            action_context_mask,
            preserve_future_mask,
            video_time_sinusoids,
            action_time_sinusoids,
            video_zero_sinusoid,
            video_deltas,
            action_deltas,
            first_frame_zero_rows: RefCell::new(None),
            video_state_a,
            video_state_b,
            video_state_c,
            action_state_a,
            action_state_b,
        })
    }

    pub(crate) fn backend(&self) -> &FastWamBackend {
        &self.backend
    }

    /// Execute the fixed official eager profile starting from VAE latents and
    /// deterministic reference noise. Image encode/decode is owned by the VAE
    /// module and intentionally remains outside this DiT method.
    pub(crate) fn infer_eager_from_latents(
        &self,
        raw_context: &Tensor,
        proprio: &Tensor,
        first_frame_latents: &Tensor,
        action_noise: &Tensor,
        video_noise: Option<&Tensor>,
    ) -> Result<FastWamEagerOutput> {
        let raw_context = raw_context.reshape(vec![self.config.context_len, self.config.text_dim])?;
        let proprio = proprio.reshape(vec![1, self.config.proprio_dim])?;
        let proprio_token = self.linear(&proprio, &self.weights.proprio_encoder)?;
        let context = ops::concat_rows(self.backend.context(), &raw_context, &proprio_token)?;
        let action = action_noise.reshape(vec![self.config.action_horizon, self.config.action_dim])?;

        match self.config.mode {
            super::FastWamMode::Fast => {
                let video = self.prepare_video(
                    first_frame_latents,
                    &self.video_zero_sinusoid,
                    &context,
                    &self.video_context_mask,
                )?;
                let cache = self.prefill_video_cache(video)?;
                let action = self.denoise_action(action, &context, &cache)?;
                Ok(FastWamEagerOutput {
                    action: action.reshape(vec![1, self.config.action_horizon, self.config.action_dim])?,
                    video_latents: None,
                })
            }
            super::FastWamMode::Idm => {
                let noise = video_noise.ok_or_else(|| {
                    Error::Other("Fast-WAM IDM requires deterministic video_noise".into())
                })?;
                let mut video = self.video_state_a.clone();
                let mut next_video = self.video_state_b.clone();
                let mut frame_video = self.video_state_c.clone();
                self.copy_state(noise, &video)?;
                let initial_mark = apxinf_musa::session_checkpoint();
                self.set_first_frame_into(&video, first_frame_latents, &next_video)?;
                if let Some(mark) = initial_mark {
                    apxinf_musa::session_rewind(mark);
                }
                std::mem::swap(&mut video, &mut next_video);
                for step in 0..self.config.num_inference_steps {
                    let mark = apxinf_musa::session_checkpoint();
                    {
                        let prepared = self.prepare_video(
                            &video,
                            &self.video_time_sinusoids[step],
                            &context,
                            &self.video_context_mask,
                        )?;
                        let flow = self.forward_video(prepared)?;
                        self.euler_update_into(
                            &video,
                            &flow,
                            &self.video_deltas[step],
                            &next_video,
                        )?;
                        self.set_first_frame_into(&next_video, first_frame_latents, &frame_video)?;
                    }
                    if let Some(mark) = mark {
                        apxinf_musa::session_rewind(mark);
                    }
                    // current <- frame-constrained output, next <- old current,
                    // frame <- old next (three fixed buffers, no aliasing).
                    std::mem::swap(&mut video, &mut frame_video);
                    std::mem::swap(&mut next_video, &mut frame_video);
                }
                let prepared = self.prepare_video(
                    &video,
                    &self.video_zero_sinusoid,
                    &context,
                    &self.video_context_mask,
                )?;
                let cache = self.prefill_video_cache(prepared)?;
                let action = self.denoise_action(action, &context, &cache)?;
                Ok(FastWamEagerOutput {
                    action: action.reshape(vec![1, self.config.action_horizon, self.config.action_dim])?,
                    video_latents: Some(video),
                })
            }
            super::FastWamMode::Joint => {
                let noise = video_noise.ok_or_else(|| {
                    Error::Other("Fast-WAM Joint requires deterministic video_noise".into())
                })?;
                let mut video = self.video_state_a.clone();
                let mut next_video = self.video_state_b.clone();
                let mut frame_video = self.video_state_c.clone();
                let mut action = self.action_state_a.clone();
                let mut next_action = self.action_state_b.clone();
                self.copy_state(noise, &video)?;
                self.copy_state(&action_noise.reshape(vec![self.config.action_horizon, self.config.action_dim])?, &action)?;
                let initial_mark = apxinf_musa::session_checkpoint();
                self.set_first_frame_into(&video, first_frame_latents, &next_video)?;
                if let Some(mark) = initial_mark {
                    apxinf_musa::session_rewind(mark);
                }
                std::mem::swap(&mut video, &mut next_video);
                for step in 0..self.config.num_inference_steps {
                    let mark = apxinf_musa::session_checkpoint();
                    {
                        let video_prepared = self.prepare_video(
                            &video,
                            &self.video_time_sinusoids[step],
                            &context,
                            &self.video_context_mask,
                        )?;
                        let action_prepared = self.prepare_action(
                            &action,
                            &self.action_time_sinusoids[step],
                            &context,
                            &self.action_context_mask,
                        )?;
                        let (video_flow, action_flow) =
                            self.forward_joint(video_prepared, action_prepared)?;
                        self.euler_update_into(
                            &video,
                            &video_flow,
                            &self.video_deltas[step],
                            &next_video,
                        )?;
                        self.set_first_frame_into(&next_video, first_frame_latents, &frame_video)?;
                        self.euler_update_into(
                            &action,
                            &action_flow,
                            &self.action_deltas[step],
                            &next_action,
                        )?;
                    }
                    if let Some(mark) = mark {
                        apxinf_musa::session_rewind(mark);
                    }
                    std::mem::swap(&mut video, &mut frame_video);
                    std::mem::swap(&mut next_video, &mut frame_video);
                    std::mem::swap(&mut action, &mut next_action);
                }
                Ok(FastWamEagerOutput {
                    action: action.reshape(vec![1, self.config.action_horizon, self.config.action_dim])?,
                    video_latents: Some(video),
                })
            }
        }
    }

    fn denoise_action(
        &self,
        action: Tensor,
        context: &Tensor,
        cache: &VideoKvCache,
    ) -> Result<Tensor> {
        let action_state = self.action_state_a.clone();
        let mut next_action = self.action_state_b.clone();
        // The source argument is the fixed input noise; copy it into the
        // persistent state before the first step.  Keeping the state outside
        // the session arena lets the step scratch be rewound safely.
        self.copy_state(&action, &action_state)?;
        let mut action = action_state;
        for step in 0..self.config.num_inference_steps {
            let mark = apxinf_musa::session_checkpoint();
            {
                let prepared = self.prepare_action(
                    &action,
                    &self.action_time_sinusoids[step],
                    context,
                    &self.action_context_mask,
                )?;
                let flow = self.forward_action_with_video_cache(prepared, cache)?;
                self.euler_update_into(
                    &action,
                    &flow,
                    &self.action_deltas[step],
                    &next_action,
                )?;
            }
            if let Some(mark) = mark {
                apxinf_musa::session_rewind(mark);
            }
            std::mem::swap(&mut action, &mut next_action);
        }
        Ok(action)
    }

    fn euler_update_into(
        &self,
        state: &Tensor,
        flow: &Tensor,
        delta: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let scaled = ops::binary_bf16(
            self.backend.context(),
            flow,
            delta,
            BinaryOp::Multiply,
        )?;
        ops::binary_bf16_into(
            self.backend.context(),
            state,
            &scaled,
            output,
            BinaryOp::Add,
        )
    }

    fn copy_state(&self, source: &Tensor, destination: &Tensor) -> Result<()> {
        apxinf_musa::transfers::copy_device_to_device(
            source,
            destination,
            self.backend.context().stream(),
        )
    }

    fn set_first_frame_into(
        &self,
        state: &Tensor,
        first_frame: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let future = ops::binary_bf16(
            self.backend.context(),
            state,
            &self.preserve_future_mask,
            BinaryOp::Multiply,
        )?;
        let [batch, channels, frames, height, width] = first_frame.shape().dims() else {
            return Err(Error::Other("Fast-WAM first frame must be NCTHW".into()));
        };
        let row_width = *batch * *channels * *height * *width;
        let rows = ops::permute_bf16(self.backend.context(), first_frame, &[2, 0, 1, 3, 4])?
            .reshape(vec![*frames, row_width])?;
        let cached_zero = self.first_frame_zero_rows.borrow().as_ref().cloned();
        let zero = if let Some(zero) = cached_zero {
            zero
        } else {
            let zero = self.backend.to_device(&Tensor::zeros(
                vec![self.config.latent_temporal - 1, row_width],
                DType::BF16,
            ))?;
            *self.first_frame_zero_rows.borrow_mut() = Some(zero.clone());
            zero
        };
        let rows = ops::concat_rows(self.backend.context(), &rows, &zero)?;
        let first = ops::permute_bf16(
            self.backend.context(),
            &rows.reshape(vec![
                self.config.latent_temporal,
                *batch,
                *channels,
                *height,
                *width,
            ])?,
            &[1, 2, 0, 3, 4],
        )?;
        let expected = output.shape().dims();
        if expected != state.shape().dims() {
            return Err(Error::Other("Fast-WAM state output shape mismatch".into()));
        }
        ops::binary_bf16_into(self.backend.context(), &future, &first, output, BinaryOp::Add)
    }

    pub(crate) fn prepare_action(
        &self,
        action: &Tensor,
        time_sinusoid: &Tensor,
        raw_context: &Tensor,
        context_mask: &Tensor,
    ) -> Result<ExpertPrepared> {
        require_matrix(action, self.config.action_horizon, self.config.action_dim, "action")?;
        require_matrix(time_sinusoid, 1, self.config.freq_dim, "action time sinusoid")?;
        let context_tokens = matrix_rows(raw_context, self.config.text_dim, "action context")?;
        require_mask(
            context_mask,
            self.config.action_horizon,
            context_tokens,
            "action context mask",
        )?;
        let weights = &self.weights.action;
        Ok(ExpertPrepared {
            tokens: self.linear(action, &weights.action_encoder)?,
            time: self.embedding(time_sinusoid, &weights.expert.time_embedding, UnaryOp::Silu)?,
            modulation: self.time_modulation(time_sinusoid, &weights.expert)?,
            context: self.embedding(
                raw_context,
                &weights.expert.text_embedding,
                UnaryOp::GeluTanh,
            )?,
            context_mask: context_mask.clone(),
        })
    }

    pub(crate) fn prepare_video(
        &self,
        latents: &Tensor,
        token_time_sinusoid: &Tensor,
        raw_context: &Tensor,
        context_mask: &Tensor,
    ) -> Result<ExpertPrepared> {
        let video_tokens = self.geometry.inference_video_tokens(self.config.mode);
        let expected_frames = video_tokens / self.geometry.tokens_per_frame;
        let expected_latents = [
            self.config.batch_size,
            self.config.latent_channels,
            expected_frames,
            self.config.latent_height,
            self.config.latent_width,
        ];
        if latents.shape().dims() != expected_latents {
            return Err(Error::Other(format!(
                "Fast-WAM video latents must be {expected_latents:?}, got {:?}",
                latents.shape().dims()
            )));
        }
        require_matrix(
            token_time_sinusoid,
            video_tokens,
            self.config.freq_dim,
            "video token time sinusoid",
        )?;
        let context_tokens = matrix_rows(raw_context, self.config.text_dim, "video context")?;
        require_mask(
            context_mask,
            video_tokens,
            context_tokens,
            "video context mask",
        )?;
        let weights = &self.weights.video;
        let patches = ops::convolution_bf16(
            self.backend.context(),
            ConvolutionArgs {
                input: latents,
                weight: &weights.patch_embedding_weight,
                bias: Some(&weights.patch_embedding_bias),
                padding: &[0, 0, 0],
                stride: &self.config.patch_size,
                dilation: &[1, 1, 1],
                groups: 1,
            },
        )?;
        let tokens = ops::permute_bf16(self.backend.context(), &patches, &[0, 2, 3, 4, 1])?
            .reshape(vec![video_tokens, self.config.video_hidden_dim])?;
        Ok(ExpertPrepared {
            tokens,
            time: self.embedding(
                token_time_sinusoid,
                &weights.expert.time_embedding,
                UnaryOp::Silu,
            )?,
            modulation: self.time_modulation(token_time_sinusoid, &weights.expert)?,
            context: self.embedding(
                raw_context,
                &weights.expert.text_embedding,
                UnaryOp::GeluTanh,
            )?,
            context_mask: context_mask.clone(),
        })
    }

    pub(crate) fn prefill_video_cache(
        &self,
        mut prepared: ExpertPrepared,
    ) -> Result<VideoKvCache> {
        let tokens = prepared.tokens.shape().dims()[0];
        if self.video_attention_mask.shape().dims() != [tokens, tokens] {
            return Err(Error::Other("Fast-WAM video prefill mask/token mismatch".into()));
        }
        let mut layers = Vec::with_capacity(self.config.num_layers);
        for block in &self.weights.video.expert.blocks {
            let io = self.build_attention_io(
                &prepared.tokens,
                &prepared.modulation,
                block,
                &self.video_ones,
                &self.video_rope_cos,
                &self.video_rope_sin,
            )?;
            let mixed = ops::masked_math_attention_bf16(
                self.backend.context(),
                &io.query,
                &io.key,
                &io.value,
                &self.video_attention_mask,
            )?;
            layers.push((io.key.clone(), io.value.clone()));
            prepared.tokens = self.apply_post(
                io,
                &mixed,
                &prepared.context,
                &prepared.context_mask,
                block,
                &self.video_ones,
            )?;
        }
        Ok(VideoKvCache {
            layers,
            video_tokens: tokens,
        })
    }

    fn forward_video(&self, mut prepared: ExpertPrepared) -> Result<Tensor> {
        for block in &self.weights.video.expert.blocks {
            let io = self.build_attention_io(
                &prepared.tokens,
                &prepared.modulation,
                block,
                &self.video_ones,
                &self.video_rope_cos,
                &self.video_rope_sin,
            )?;
            let mixed = ops::masked_math_attention_bf16(
                self.backend.context(),
                &io.query,
                &io.key,
                &io.value,
                &self.video_attention_mask,
            )?;
            prepared.tokens = self.apply_post(
                io,
                &mixed,
                &prepared.context,
                &prepared.context_mask,
                block,
                &self.video_ones,
            )?;
        }
        self.video_head(&prepared.tokens, &prepared.time)
    }

    pub(crate) fn forward_action_with_video_cache(
        &self,
        mut prepared: ExpertPrepared,
        cache: &VideoKvCache,
    ) -> Result<Tensor> {
        let action_tokens = prepared.tokens.shape().dims()[0];
        let total_tokens = cache.video_tokens + action_tokens;
        let action_mask = ops::contiguous_rows(
            self.backend.context(),
            &self.mot_attention_mask,
            cache.video_tokens,
            action_tokens,
        )?;
        if action_mask.shape().dims() != [action_tokens, total_tokens]
            || cache.layers.len() != self.config.num_layers
        {
            return Err(Error::Other("Fast-WAM action cache geometry mismatch".into()));
        }
        for (layer, block) in self.weights.action.expert.blocks.iter().enumerate() {
            let io = self.build_attention_io(
                &prepared.tokens,
                &prepared.modulation,
                block,
                &self.action_ones,
                &self.action_rope_cos,
                &self.action_rope_sin,
            )?;
            let key = concat_attention_rows(self.backend.context(), &cache.layers[layer].0, &io.key)?;
            let value = concat_attention_rows(
                self.backend.context(),
                &cache.layers[layer].1,
                &io.value,
            )?;
            let mixed = ops::masked_math_attention_bf16(
                self.backend.context(),
                &io.query,
                &key,
                &value,
                &action_mask,
            )?;
            prepared.tokens = self.apply_post(
                io,
                &mixed,
                &prepared.context,
                &prepared.context_mask,
                block,
                &self.action_ones,
            )?;
        }
        self.linear(&prepared.tokens, &self.weights.action.head)
    }

    pub(crate) fn forward_joint(
        &self,
        mut video: ExpertPrepared,
        mut action: ExpertPrepared,
    ) -> Result<(Tensor, Tensor)> {
        let video_tokens = video.tokens.shape().dims()[0];
        let action_tokens = action.tokens.shape().dims()[0];
        if self.mot_attention_mask.shape().dims()
            != [video_tokens + action_tokens, video_tokens + action_tokens]
        {
            return Err(Error::Other("Fast-WAM joint mask/token mismatch".into()));
        }
        for layer in 0..self.config.num_layers {
            let video_block = &self.weights.video.expert.blocks[layer];
            let action_block = &self.weights.action.expert.blocks[layer];
            let video_io = self.build_attention_io(
                &video.tokens,
                &video.modulation,
                video_block,
                &self.video_ones,
                &self.video_rope_cos,
                &self.video_rope_sin,
            )?;
            let action_io = self.build_attention_io(
                &action.tokens,
                &action.modulation,
                action_block,
                &self.action_ones,
                &self.action_rope_cos,
                &self.action_rope_sin,
            )?;
            let query = concat_attention_rows(self.backend.context(), &video_io.query, &action_io.query)?;
            let key = concat_attention_rows(self.backend.context(), &video_io.key, &action_io.key)?;
            let value = concat_attention_rows(self.backend.context(), &video_io.value, &action_io.value)?;
            let mixed = ops::masked_math_attention_bf16(
                self.backend.context(),
                &query,
                &key,
                &value,
                &self.mot_attention_mask,
            )?;
            let mixed_rows = ops::permute_bf16(self.backend.context(), &mixed, &[0, 2, 1, 3])?
                .reshape(vec![video_tokens + action_tokens, self.config.num_heads * self.config.head_dim])?;
            let mixed_video = ops::contiguous_rows(
                self.backend.context(),
                &mixed_rows,
                0,
                video_tokens,
            )?
            .reshape(vec![1, video_tokens, self.config.num_heads, self.config.head_dim])?;
            let mixed_video = ops::permute_bf16(self.backend.context(), &mixed_video, &[0, 2, 1, 3])?;
            let mixed_action = ops::contiguous_rows(
                self.backend.context(),
                &mixed_rows,
                video_tokens,
                action_tokens,
            )?
            .reshape(vec![1, action_tokens, self.config.num_heads, self.config.head_dim])?;
            let mixed_action = ops::permute_bf16(self.backend.context(), &mixed_action, &[0, 2, 1, 3])?;
            video.tokens = self.apply_post(
                video_io,
                &mixed_video,
                &video.context,
                &video.context_mask,
                video_block,
                &self.video_ones,
            )?;
            action.tokens = self.apply_post(
                action_io,
                &mixed_action,
                &action.context,
                &action.context_mask,
                action_block,
                &self.action_ones,
            )?;
        }
        Ok((
            self.video_head(&video.tokens, &video.time)?,
            self.linear(&action.tokens, &self.weights.action.head)?,
        ))
    }

    fn build_attention_io(
        &self,
        tokens: &Tensor,
        time_modulation: &Tensor,
        block: &DiTBlockWeights,
        ones: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<ExpertAttentionIo> {
        let rows = tokens.shape().dims()[0];
        let hidden = tokens.shape().dims()[1];
        let mod_rows = time_modulation.shape().dims()[0];
        if time_modulation.shape().dims() != [mod_rows, 6, hidden] {
            return Err(Error::Other(format!(
                "Fast-WAM t_mod must be [M,6,D], got {:?}",
                time_modulation.shape().dims()
            )));
        }
        let modulation = ops::binary_bf16(
            self.backend.context(),
            time_modulation,
            &block.modulation,
            BinaryOp::Add,
        )?;
        let modulation = ops::permute_bf16(self.backend.context(), &modulation, &[1, 0, 2])?
            .reshape(vec![6 * mod_rows, hidden])?;
        let part = |index| {
            ops::contiguous_rows(
                self.backend.context(),
                &modulation,
                index * mod_rows,
                mod_rows,
            )
        };
        let shift_msa = part(0)?;
        let scale_msa = part(1)?;
        let gate_msa = part(2)?;
        let shift_mlp = part(3)?;
        let scale_mlp = part(4)?;
        let gate_mlp = part(5)?;
        if mod_rows != 1 && mod_rows != rows {
            return Err(Error::Other(format!(
                "Fast-WAM t_mod rows must be 1 or token count {rows}, got {mod_rows}"
            )));
        }
        let normalized = ops::layer_norm_bf16(
            self.backend.context(),
            tokens,
            None,
            None,
            f64::from(self.config.norm_eps),
        )?;
        let attention_input = self.modulate(&normalized, &shift_msa, &scale_msa, ones)?;
        let query = self.attention_projection(
            &attention_input,
            &block.self_attention.query,
            &block.self_attention.query_norm,
            rope_cos,
            rope_sin,
        )?;
        let key = self.attention_projection(
            &attention_input,
            &block.self_attention.key,
            &block.self_attention.key_norm,
            rope_cos,
            rope_sin,
        )?;
        let value = self
            .linear(&attention_input, &block.self_attention.value)?
            .reshape(vec![1, rows, self.config.num_heads, self.config.head_dim])?;
        let value = ops::permute_bf16(self.backend.context(), &value, &[0, 2, 1, 3])?;
        Ok(ExpertAttentionIo {
            query,
            key,
            value,
            residual: tokens.clone(),
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        })
    }

    fn apply_post(
        &self,
        io: ExpertAttentionIo,
        mixed: &Tensor,
        context: &Tensor,
        context_mask: &Tensor,
        block: &DiTBlockWeights,
        ones: &Tensor,
    ) -> Result<Tensor> {
        let rows = io.residual.shape().dims()[0];
        let attention_width = self.config.num_heads * self.config.head_dim;
        let mixed = ops::permute_bf16(self.backend.context(), mixed, &[0, 2, 1, 3])?;
        let projected = self.linear(
            &mixed.reshape(vec![rows, attention_width])?,
            &block.self_attention.output,
        )?;
        let gated = ops::binary_bf16(
            self.backend.context(),
            &projected,
            &io.gate_msa,
            BinaryOp::Multiply,
        )?;
        let mut hidden = ops::binary_bf16(
            self.backend.context(),
            &io.residual,
            &gated,
            BinaryOp::Add,
        )?;
        let cross_input = ops::layer_norm_bf16(
            self.backend.context(),
            &hidden,
            Some(&block.cross_attention_input_norm.weight),
            Some(&block.cross_attention_input_norm.bias),
            f64::from(self.config.norm_eps),
        )?;
        let cross = self.cross_attention(
            &cross_input,
            context,
            context_mask,
            &block.cross_attention,
        )?;
        hidden = ops::binary_bf16(
            self.backend.context(),
            &hidden,
            &cross,
            BinaryOp::Add,
        )?;
        let mlp_norm = ops::layer_norm_bf16(
            self.backend.context(),
            &hidden,
            None,
            None,
            f64::from(self.config.norm_eps),
        )?;
        let mlp_input = self.modulate(&mlp_norm, &io.shift_mlp, &io.scale_mlp, ones)?;
        let mlp = self.linear(&mlp_input, &block.feed_forward_input)?;
        let mlp = ops::unary_bf16(self.backend.context(), &mlp, UnaryOp::GeluTanh)?;
        let mlp = self.linear(&mlp, &block.feed_forward_output)?;
        let mlp = ops::binary_bf16(
            self.backend.context(),
            &mlp,
            &io.gate_mlp,
            BinaryOp::Multiply,
        )?;
        ops::binary_bf16(self.backend.context(), &hidden, &mlp, BinaryOp::Add)
    }

    fn cross_attention(
        &self,
        input: &Tensor,
        context: &Tensor,
        mask: &Tensor,
        weights: &AttentionWeights,
    ) -> Result<Tensor> {
        let query_tokens = input.shape().dims()[0];
        let context_tokens = context.shape().dims()[0];
        require_mask(mask, query_tokens, context_tokens, "cross-attention mask")?;
        let query = self.normalized_attention_projection(input, &weights.query, &weights.query_norm)?;
        let key = self.normalized_attention_projection(context, &weights.key, &weights.key_norm)?;
        let value = self
            .linear(context, &weights.value)?
            .reshape(vec![1, context_tokens, self.config.num_heads, self.config.head_dim])?;
        let value = ops::permute_bf16(self.backend.context(), &value, &[0, 2, 1, 3])?;
        let mixed = ops::masked_math_attention_bf16(
            self.backend.context(),
            &query,
            &key,
            &value,
            mask,
        )?;
        self.linear(
            &ops::permute_bf16(self.backend.context(), &mixed, &[0, 2, 1, 3])?.reshape(vec![
                query_tokens,
                self.config.num_heads * self.config.head_dim,
            ])?,
            &weights.output,
        )
    }

    fn normalized_attention_projection(
        &self,
        input: &Tensor,
        linear: &LinearWeights,
        norm_weight: &Tensor,
    ) -> Result<Tensor> {
        let rows = input.shape().dims()[0];
        let projection = self.linear(input, linear)?;
        let projection = ops::rms_norm_bf16(
            self.backend.context(),
            &projection,
            f64::from(self.config.norm_eps),
        )?;
        let projected = ops::binary_bf16(
            self.backend.context(),
            &projection,
            norm_weight,
            BinaryOp::Multiply,
        )?
        .reshape(vec![1, rows, self.config.num_heads, self.config.head_dim])?;
        ops::permute_bf16(self.backend.context(), &projected, &[0, 2, 1, 3])
    }

    fn attention_projection(
        &self,
        input: &Tensor,
        linear: &LinearWeights,
        norm_weight: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let projection = self.normalized_attention_projection(input, linear, norm_weight)?;
        ops::pairwise_rope_bf16(self.backend.context(), &projection, rope_cos, rope_sin)
    }

    fn time_modulation(&self, sinusoid: &Tensor, weights: &ExpertWeights) -> Result<Tensor> {
        let time = self.embedding(sinusoid, &weights.time_embedding, UnaryOp::Silu)?;
        let time = ops::unary_bf16(self.backend.context(), &time, UnaryOp::Silu)?;
        let rows = time.shape().dims()[0];
        self.linear(&time, &weights.time_projection)?
            .reshape(vec![rows, 6, time.shape().dims()[1]])
    }

    fn embedding(
        &self,
        input: &Tensor,
        weights: &EmbeddingWeights,
        activation: UnaryOp,
    ) -> Result<Tensor> {
        let hidden = self.linear(input, &weights.input)?;
        let hidden = ops::unary_bf16(self.backend.context(), &hidden, activation)?;
        self.linear(&hidden, &weights.output)
    }

    fn linear(&self, input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
        self.backend.linear(input, &weights.weight, &weights.bias)
    }

    fn modulate(
        &self,
        input: &Tensor,
        shift: &Tensor,
        scale: &Tensor,
        ones: &Tensor,
    ) -> Result<Tensor> {
        let scale = ops::binary_bf16(
            self.backend.context(),
            scale,
            ones,
            BinaryOp::Add,
        )?;
        let scaled = ops::binary_bf16(
            self.backend.context(),
            input,
            &scale,
            BinaryOp::Multiply,
        )?;
        ops::binary_bf16(self.backend.context(), &scaled, shift, BinaryOp::Add)
    }

    fn video_head(&self, tokens: &Tensor, time: &Tensor) -> Result<Tensor> {
        let rows = tokens.shape().dims()[0];
        if time.shape().dims() != [rows, self.config.video_hidden_dim] {
            return Err(Error::Other("Fast-WAM video head time/token mismatch".into()));
        }
        let modulation = ops::binary_bf16(
            self.backend.context(),
            &time.reshape(vec![rows, 1, self.config.video_hidden_dim])?,
            &self.weights.video.head_modulation,
            BinaryOp::Add,
        )?;
        let modulation = ops::permute_bf16(self.backend.context(), &modulation, &[1, 0, 2])?
            .reshape(vec![2 * rows, self.config.video_hidden_dim])?;
        let shift = ops::contiguous_rows(self.backend.context(), &modulation, 0, rows)?;
        let scale = ops::contiguous_rows(self.backend.context(), &modulation, rows, rows)?;
        let normalized = ops::layer_norm_bf16(
            self.backend.context(),
            tokens,
            None,
            None,
            f64::from(self.config.norm_eps),
        )?;
        let hidden = self.modulate(&normalized, &shift, &scale, &self.video_ones)?;
        let patches = self.linear(&hidden, &self.weights.video.head)?;
        let [pt, ph, pw] = self.config.patch_size;
        let [frames, height, width] = self.geometry.video_grid;
        ops::permute_bf16(
            self.backend.context(),
            &patches.reshape(vec![
                1,
                frames,
                height,
                width,
                pt,
                ph,
                pw,
                self.config.latent_channels,
            ])?,
            &[0, 7, 1, 4, 2, 5, 3, 6],
        )?
        .reshape(vec![
            1,
            self.config.latent_channels,
            frames * pt,
            height * ph,
            width * pw,
        ])
    }
}

fn concat_attention_rows(
    ctx: &apxinf_musa::MusaContext,
    first: &Tensor,
    second: &Tensor,
) -> Result<Tensor> {
    let first_dims = first.shape().dims();
    let second_dims = second.shape().dims();
    let [1, heads, first_tokens, head_dim] = first_dims else {
        return Err(Error::Other(format!(
            "Fast-WAM attention concat expects first [1,H,T,D], got {first_dims:?}"
        )));
    };
    let [1, second_heads, second_tokens, second_dim] = second_dims else {
        return Err(Error::Other(format!(
            "Fast-WAM attention concat expects second [1,H,T,D], got {second_dims:?}"
        )));
    };
    if heads != second_heads || head_dim != second_dim {
        return Err(Error::Other("Fast-WAM attention concat head mismatch".into()));
    }
    let first = ops::permute_bf16(ctx, first, &[0, 2, 1, 3])?
        .reshape(vec![*first_tokens, heads * head_dim])?;
    let second = ops::permute_bf16(ctx, second, &[0, 2, 1, 3])?
        .reshape(vec![*second_tokens, heads * head_dim])?;
    let rows = ops::concat_rows(
        ctx,
        &first,
        &second,
    )?;
    let rows = rows.reshape(vec![1, first_tokens + second_tokens, *heads, *head_dim])?;
    ops::permute_bf16(ctx, &rows, &[0, 2, 1, 3])
}

fn upload_rope(backend: &FastWamBackend, rope: &RopeTable, cosine: bool) -> Result<Tensor> {
    let values = if cosine { rope.cos() } else { rope.sin() };
    backend.to_device(&Tensor::from_f32(
        vec![rope.tokens(), rope.pairs_per_head()],
        values,
    )?)
}

fn upload_ones(backend: &FastWamBackend, width: usize) -> Result<Tensor> {
    backend.to_device(&Tensor::from_bf16(
        vec![width],
        &vec![half::bf16::ONE; width],
    )?)
}

fn upload_true_mask(backend: &FastWamBackend, rows: usize, cols: usize) -> Result<Tensor> {
    backend.to_device(&Tensor::from_i8(vec![rows, cols], &vec![1; rows * cols])?)
}

fn upload_sinusoid(backend: &FastWamBackend, dim: usize, positions: &[f32]) -> Result<Tensor> {
    let values = sinusoidal_embedding_1d(dim, positions)?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_bf16(vec![positions.len(), dim], &values)?)
}

fn upload_scalars(backend: &FastWamBackend, values: &[f32]) -> Result<Vec<Tensor>> {
    values
        .iter()
        .map(|&value| {
            backend.to_device(&Tensor::from_bf16(
                vec![1],
                &[half::bf16::from_f32(value)],
            )?)
        })
        .collect()
}

fn require_matrix(tensor: &Tensor, rows: usize, cols: usize, name: &str) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != [rows, cols] {
        return Err(Error::Other(format!(
            "Fast-WAM {name} must be BF16 [{rows},{cols}], got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn matrix_rows(tensor: &Tensor, cols: usize, name: &str) -> Result<usize> {
    let [rows, actual_cols] = tensor.shape().dims() else {
        return Err(Error::Other(format!(
            "Fast-WAM {name} must be a matrix, got {:?}",
            tensor.shape().dims()
        )));
    };
    if tensor.dtype() != DType::BF16 || *actual_cols != cols {
        return Err(Error::Other(format!(
            "Fast-WAM {name} must be BF16 [L,{cols}], got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(*rows)
}

fn require_mask(mask: &Tensor, rows: usize, cols: usize, name: &str) -> Result<()> {
    if mask.dtype() != DType::I8 || mask.shape().dims() != [rows, cols] {
        return Err(Error::Other(format!(
            "Fast-WAM {name} must be I8 [{rows},{cols}], got {} {:?}",
            mask.dtype(),
            mask.shape().dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_input_contracts_reject_wrong_action_width() {
        let action = Tensor::zeros(vec![32, 8], DType::BF16);
        assert!(require_matrix(&action, 32, 7, "action")
            .unwrap_err()
            .to_string()
            .contains("[32,7]"));
    }

    #[test]
    fn rectangular_context_mask_contract_is_explicit() {
        let mask = Tensor::zeros(vec![32, 129], DType::I8);
        require_mask(&mask, 32, 129, "action context mask").unwrap();
        assert!(require_mask(&mask, 98, 129, "video context mask").is_err());
    }
}
