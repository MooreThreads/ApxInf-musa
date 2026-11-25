//! Shared PI0.5 preparation and captured resource ownership.
use crate::musa_pi05::backend::{
    self, capture as capture_cuda_graph, ops, transfers, Context, DeviceBuffer, ExecutionSession,
};
use crate::musa_pi05::model::Pi05Model;
use crate::musa_pi05::model::{ModelOperation, ModelVariant, PrepareBlocks};
use crate::musa_pi05::{Pi05Config, Pi05ImageLayout};
use apxinf_core::{Error, Graph, Result, Tensor};
use std::sync::Arc;

/// Smallest useful first rung. Below this the ladder would add plan attempts
/// without meaningfully reducing the reservation.
const WORKSPACE_FLOOR: usize = 256 << 20;

/// Capacities the capture and eager paths try in order.
///
/// The first rung is an eighth of the modeled reservation, floored at
/// [`WORKSPACE_FLOOR`], doubling until the proven reservation. A bump allocator
/// never reuses a slice, so every rung below the full reservation would
/// exhaust: `POOL=0` gets the full reservation directly.
pub(super) fn workspace_capacities(required_bytes: usize, pooling: bool) -> Vec<usize> {
    if !pooling || required_bytes <= WORKSPACE_FLOOR {
        return vec![required_bytes];
    }
    let mut capacities = Vec::new();
    let mut capacity = (required_bytes / 8).max(WORKSPACE_FLOOR);
    while capacity < required_bytes {
        capacities.push(capacity);
        capacity = match capacity.checked_mul(2) {
            Some(next) => next,
            None => break,
        };
    }
    if capacities.last().copied() != Some(required_bytes) {
        capacities.push(required_bytes);
    }
    capacities
}

/// One line per successful plan, so the allocator decision is auditable.
pub(super) fn report_workspace(
    session: &ExecutionSession,
    attempts: usize,
    token_count: usize,
    label: &str,
) {
    if !workspace_report_enabled() {
        return;
    }
    let workspace = session.workspace();
    eprintln!(
        "[apxinf] PI0.5 {label} workspace: capacity={} used={} peak={} attempts={} tokens={} pool={}",
        workspace.capacity(),
        workspace.used(),
        workspace.peak_used(),
        attempts,
        token_count,
        workspace.pooling(),
    );
}

fn workspace_report_enabled() -> bool {
    static REPORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REPORT.get_or_init(|| std::env::var("APXINF_MUSA_WORKSPACE_REPORT").as_deref() == Ok("1"))
}

/// Run `prepare` on each arena capacity until one fits, then let `finish`
/// commit the winning rung.
///
/// Every rung gets a fresh session. `prepare` is the only step the ladder
/// retries: an error the arena reports as exhaustion is ordered (the failed
/// traversal may still have work in flight), the session dropped, and the next
/// rung tried. Any other prepare error is final.
///
/// `finish` runs once, on the rung that prepared, and is never retried: a
/// capture that fails after a fitting warm-up would repeat the traversal on the
/// next rung for an error capacity cannot fix. The report is printed after
/// `finish`, so it describes the plan that actually exists.
pub(super) fn with_capacity_ladder<P, T>(
    context: &Context,
    label: &'static str,
    required: usize,
    token_count: usize,
    mut prepare: impl FnMut(&ExecutionSession) -> Result<P>,
    finish: impl FnOnce(&ExecutionSession, P) -> Result<T>,
) -> Result<(T, ExecutionSession)> {
    let capacities = workspace_capacities(required, backend::workspace_pool_enabled());
    let mut last_exhaustion = None;
    for (attempt, &capacity) in capacities.iter().enumerate() {
        let session = ExecutionSession::with_capacity(capacity, context.device_id())?;
        let prepared = match prepare(&session) {
            Ok(prepared) => prepared,
            Err(error) if backend::is_workspace_exhaustion(&error) => {
                last_exhaustion = Some(error);
                // The failed traversal may still have work in flight; order it
                // before releasing the arena it wrote into.
                backend::synchronize(context)?;
                drop(session);
                continue;
            }
            Err(error) => return Err(error),
        };
        let output = finish(&session, prepared)?;
        report_workspace(&session, attempt + 1, token_count, label);
        return Ok((output, session));
    }
    Err(ladder_exhaustion_error(
        label,
        required,
        capacities.len(),
        last_exhaustion.as_ref(),
    ))
}

fn ladder_exhaustion_error(
    label: &str,
    required: usize,
    attempts: usize,
    last: Option<&Error>,
) -> Error {
    Error::Other(format!(
        "PI0.5 {label} workspace exhausted at {required} bytes after {attempts} attempts: {}",
        last.map(|error| error.to_string())
            .unwrap_or_else(|| "unknown error".into())
    ))
}

pub struct CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<DeviceBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    token_ids: DeviceBuffer,
    token_count: usize,
    backend: Arc<Context>,
    // Retains every fixed resource the recorded computation reads: the weights,
    // and the model's precomputed modulation, which lives on the model. Dropping
    // this before the graph would free addresses the graph still captures.
    _fixed: Box<dyn std::any::Any>,
    session: ExecutionSession,
}

impl CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn replay_and_synchronize(&self) -> Result<()> {
        self.graph.replay()?;
        backend::synchronize(&self.backend)
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn raw_image_layout(&self) -> Option<Pi05ImageLayout> {
        self.raw_image_layout
    }

    fn update_tokens(&self, token_ids: &[u32]) -> Result<()> {
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Other)
    }

    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_device(noise, &self.noise)
    }

    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "PI0.5 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "PI0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        backend::synchronize(&self.backend)?;
        transfers::copy_cpu_to_device(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    pub fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<()> {
        self.update_raw_image_inputs_without_noise(images, token_ids)?;
        transfers::copy_cpu_to_device(noise, &self.noise)
    }

    pub fn update_raw_image_inputs_without_noise(
        &self,
        images: &[u8],
        token_ids: &[u32],
    ) -> Result<()> {
        let raw_images = self.raw_images.as_ref().ok_or_else(|| {
            Error::Other("PI0.5 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "PI0.5 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "PI0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        backend::synchronize(&self.backend)?;
        raw_images.copy_from_host(images).map_err(Error::Other)?;
        self.update_tokens(token_ids)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.session.workspace().capacity()
    }

    pub fn workspace_used_bytes(&self) -> usize {
        self.session.workspace().used()
    }

    /// The GEMM and attention shapes the warm-up traversal ran, in first-use
    /// order.
    ///
    /// This is the list the captured graph is a record of. Capture can only
    /// bake shapes whose provider state already exists, so this listing being
    /// complete for the model is what makes the graph legal, and a shape
    /// outside it is what the session refuses by name.
    pub fn prepared_shapes(&self) -> Vec<backend::OperationShape> {
        self.session.prepared_shapes()
    }
}

struct CaptureBuilder<'a, B: PrepareBlocks> {
    model: &'a Arc<Pi05Model<B>>,
    backend: &'a Arc<Context>,
    config: &'a Pi05Config,
}
impl<B: PrepareBlocks> CaptureBuilder<'_, B> {
    /// The captured graph retains the context, so this hands out the `Arc`
    /// rather than a borrow of its contents.
    fn ctx(&self) -> &Arc<Context> {
        &self.backend
    }

    /// One traversal, reading the modulation the model built when it was
    /// assembled.
    ///
    /// Every tensor the graph reads is one of four kinds: a weight, one of the
    /// caller's input buffers, an arena slice this traversal wrote, or the
    /// modulation — a driver allocation the model owns for its whole life.
    /// Ownership is the distinction that matters, not where the memory came
    /// from: built as a temporary before capture, the same allocation is freed
    /// and the arena reserved next takes its addresses, which is what once made
    /// a replay read another traversal's outputs.
    ///
    /// Nothing here computes modulation, and nothing here can: the traversal
    /// only sees the model, and the model was given its modulation at
    /// construction.
    fn infer_captured_inputs(
        &self,
        patches: &Tensor,
        raw_images: Option<&DeviceBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => self.model.infer(patches, token_ids, token_count, noise),
            (Some(images), Some(layout)) => {
                self.model.preprocess(images, patches, layout)?;
                self.model
                    .infer_native(patches, token_ids, token_count, noise)
            }
            _ => Err(Error::Other(
                "PI0.5 raw image capture state is inconsistent".into(),
            )),
        }
    }

    fn capture_infer_impl(
        &self,
        patches: Tensor,
        raw_images: Option<DeviceBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<CapturedGraph> {
        let backend = self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "PI0.5 raw image capture state is inconsistent".into(),
            ));
        }
        backend::synchronize(backend)?;
        let required = self.model.workspace_requirements(token_count)?.bytes;
        let ((graph, output), session) = with_capacity_ladder(
            self.ctx(),
            "graph",
            required,
            token_count,
            |session| {
                // Warm-up and capture resolve providers under the same rule. Left on,
                // autotuning would measure the warm-up's shapes and change the provider
                // between the two traversals — the record capture bakes would then be
                // of a dispatch that no longer happens.
                backend::without_autotune(|| {
                    ops::prepare_with_session(session, || {
                        self.infer_captured_inputs(
                            &patches,
                            raw_images.as_ref(),
                            raw_image_layout,
                            token_ids,
                            token_count,
                            noise,
                        )
                    })
                })
            },
            |session, eager_output| {
                backend::synchronize(backend)?;

                let captured_output = std::cell::RefCell::new(None);
                let graph = backend::without_autotune(|| {
                    capture_cuda_graph(self.ctx(), || {
                        let output = ops::with_session(session, || {
                            self.infer_captured_inputs(
                                &patches,
                                raw_images.as_ref(),
                                raw_image_layout,
                                token_ids,
                                token_count,
                                noise,
                            )
                        })?;
                        captured_output.replace(Some(output));
                        Ok(())
                    })
                })?;
                drop(eager_output);
                let output = captured_output.into_inner().ok_or_else(|| {
                    Error::Other("PI0.5 graph capture produced no output tensor".into())
                })?;
                let graph: Box<dyn Graph> = Box::new(graph);
                Ok((graph, output))
            },
        )?;
        Ok(CapturedGraph {
            graph,
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _fixed: Box::new(Arc::clone(&self.model)),
            session,
        })
    }

    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<CapturedGraph> {
        self.capture_infer_impl(patches.clone(), None, None, token_ids, token_count, noise)
    }

    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes = self
            .config
            .num_views
            .checked_mul(3)
            .and_then(|n| n.checked_mul(self.config.image_size))
            .and_then(|n| n.checked_mul(self.config.image_size))
            .ok_or_else(|| Error::Other("PI0.5 raw image size overflow".into()))?;
        let raw_images = DeviceBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Other)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches = backend::to_device(
            backend,
            &Tensor::zeros(
                vec![patch_rows, patch_width],
                crate::musa_pi05::model::ACTIVATION_DTYPE,
            ),
        )?;
        self.capture_infer_impl(
            patches,
            Some(raw_images),
            Some(layout),
            token_ids,
            token_count,
            noise,
        )
    }
}

pub(super) enum CaptureInput<'a> {
    Patches(&'a Tensor),
    Rgb(Pi05ImageLayout),
}
pub(super) fn capture<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    input: CaptureInput<'_>,
    tokens: &DeviceBuffer,
    count: usize,
    noise: &Tensor,
) -> Result<CapturedGraph> {
    let builder = CaptureBuilder {
        model,
        backend: model.backend(),
        config: model.config(),
    };
    match input {
        CaptureInput::Rgb(layout) => builder.capture_infer_rgb_u8(layout, tokens, count, noise),
        CaptureInput::Patches(patches) => builder.capture_infer(patches, tokens, count, noise),
    }
}

pub(super) fn capture_loaded(
    model: &ModelVariant,
    spec: &crate::vla::InferenceSpec,
    patches: &Tensor,
    tokens: &DeviceBuffer,
    noise: &Tensor,
) -> Result<CapturedGraph> {
    struct CaptureOperation<'a> {
        input: CaptureInput<'a>,
        tokens: &'a DeviceBuffer,
        count: usize,
        noise: &'a Tensor,
    }
    impl ModelOperation for CaptureOperation<'_> {
        type Output = CapturedGraph;
        fn run<B: PrepareBlocks>(self, model: &Arc<Pi05Model<B>>) -> Result<Self::Output> {
            capture(model, self.input, self.tokens, self.count, self.noise)
        }
    }
    let input = match spec.image_layout {
        Some(layout) => CaptureInput::Rgb(super::runner::kernel_image_layout(layout)),
        None => CaptureInput::Patches(patches),
    };
    model.with_model(CaptureOperation {
        input,
        tokens,
        count: spec.token_count,
        noise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capacity_ladder_is_ascending_and_ends_at_the_reservation() {
        let required = 5_909_589_760;
        assert_eq!(
            workspace_capacities(required, true),
            vec![738_698_720, 1_477_397_440, 2_954_794_880, 5_909_589_760]
        );
    }

    #[test]
    fn a_reservation_below_the_floor_is_its_own_capacity() {
        assert_eq!(workspace_capacities(1 << 20, true), vec![1 << 20]);
    }

    #[test]
    fn a_reservation_at_the_floor_is_a_single_rung() {
        assert_eq!(
            workspace_capacities(WORKSPACE_FLOOR, true),
            vec![WORKSPACE_FLOOR]
        );
    }

    #[test]
    fn a_reservation_just_above_the_floor_starts_at_the_floor() {
        assert_eq!(
            workspace_capacities(WORKSPACE_FLOOR + 1, true),
            vec![WORKSPACE_FLOOR, WORKSPACE_FLOOR + 1]
        );
    }

    #[test]
    fn the_ladder_floors_its_first_rung_at_256_mib() {
        assert_eq!(
            workspace_capacities(1 << 30, true),
            vec![WORKSPACE_FLOOR, 512 << 20, 1 << 30]
        );
    }

    #[test]
    fn a_bump_allocator_uses_the_full_reservation() {
        let required = 5_909_589_760;
        assert_eq!(workspace_capacities(required, false), vec![required]);
        assert_eq!(workspace_capacities(1 << 20, false), vec![1 << 20]);
    }

    #[test]
    fn the_aggregate_exhaustion_error_names_the_label_and_attempts() {
        let unknown = ladder_exhaustion_error("eager", 12_345, 3, None);
        assert_eq!(
            unknown.to_string(),
            "PI0.5 eager workspace exhausted at 12345 bytes after 3 attempts: unknown error"
        );
        let last = Error::Other("static inference workspace exhausted: need 1 bytes".into());
        let named = ladder_exhaustion_error("graph", 1, 1, Some(&last));
        assert!(
            named.to_string().ends_with(
                "at 1 bytes after 1 attempts: static inference workspace exhausted: need 1 bytes"
            ),
            "{named}"
        );
    }
}
