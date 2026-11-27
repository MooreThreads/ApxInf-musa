//! Fixed token geometry and embedding tables for the first MUSA Fast-WAM profile.
//!
//! The formulas in this module follow Fast-WAM revision
//! `7faa71108368fbb3b6885649f112af607427a2d4`:
//! `wan_video_dit.py::{sinusoidal_embedding_1d,precompute_freqs_cis,
//! precompute_freqs_cis_3d,WanModel::get_freqs,build_video_to_video_mask}` and
//! `fastwam{,_joint}.py::_build_mot_attention_mask`.

use apxinf_core::{Error, Result};

use super::{FastWamConfig, FastWamMode};

const ROPE_THETA: f64 = 10_000.0;

/// A row-major boolean attention mask represented exactly as the I8 bytes
/// accepted by `apxinf_musa::ops::masked_math_attention_bf16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AttentionMask {
    rows: usize,
    cols: usize,
    values: Vec<i8>,
}

impl AttentionMask {
    fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Other("Fast-WAM attention mask size overflow".into()))?;
        Ok(Self {
            rows,
            cols,
            values: vec![0; len],
        })
    }

    fn fill_rect(&mut self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) {
        for row in row_start..row_end {
            self.values[row * self.cols + col_start..row * self.cols + col_end].fill(1);
        }
    }

    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn values(&self) -> &[i8] {
        &self.values
    }

    #[cfg(test)]
    fn allows(&self, query: usize, key: usize) -> bool {
        self.values[query * self.cols + key] != 0
    }
}

/// Cosine/sine pairs in the order consumed by Fast-WAM's complex-pair RoPE.
/// Each table is row-major `[tokens, head_dim / 2]`.
#[derive(Clone, Debug)]
pub(crate) struct RopeTable {
    tokens: usize,
    pairs_per_head: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    pub(crate) const fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) const fn pairs_per_head(&self) -> usize {
        self.pairs_per_head
    }

    pub(crate) fn cos(&self) -> &[f32] {
        &self.cos
    }

    pub(crate) fn sin(&self) -> &[f32] {
        &self.sin
    }
}

/// Derived geometry for the admitted LIBERO two-camera checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FastWamGeometry {
    pub(crate) video_grid: [usize; 3],
    pub(crate) tokens_per_frame: usize,
    pub(crate) video_tokens: usize,
    pub(crate) action_tokens: usize,
    pub(crate) head_dim: usize,
}

impl FastWamGeometry {
    pub(crate) fn new(config: &FastWamConfig) -> Result<Self> {
        config.validate()?;
        let [patch_t, patch_h, patch_w] = config.patch_size;
        if config.latent_temporal % patch_t != 0
            || config.latent_height % patch_h != 0
            || config.latent_width % patch_w != 0
        {
            return Err(Error::Other(format!(
                "Fast-WAM latent geometry {:?} is not divisible by patch {:?}",
                [
                    config.latent_temporal,
                    config.latent_height,
                    config.latent_width
                ],
                config.patch_size
            )));
        }
        let video_grid = [
            config.latent_temporal / patch_t,
            config.latent_height / patch_h,
            config.latent_width / patch_w,
        ];
        let tokens_per_frame = video_grid[1]
            .checked_mul(video_grid[2])
            .ok_or_else(|| Error::Other("Fast-WAM tokens-per-frame overflow".into()))?;
        let video_tokens = video_grid[0]
            .checked_mul(tokens_per_frame)
            .ok_or_else(|| Error::Other("Fast-WAM video-token count overflow".into()))?;
        Ok(Self {
            video_grid,
            tokens_per_frame,
            video_tokens,
            action_tokens: config.action_horizon,
            head_dim: config.head_dim,
        })
    }

    /// Fast pre-fills only the encoded observation frame. IDM and Joint use
    /// the complete three-frame latent sequence during their action stage.
    pub(crate) const fn inference_video_tokens(&self, mode: FastWamMode) -> usize {
        match mode {
            FastWamMode::Fast => self.tokens_per_frame,
            FastWamMode::Idm | FastWamMode::Joint => self.video_tokens,
        }
    }

    /// Official `first_frame_causal`: the observation-frame queries cannot
    /// read future-frame keys; every later query can read every video key.
    pub(crate) fn video_attention_mask(&self, video_tokens: usize) -> Result<AttentionMask> {
        if video_tokens == 0
            || video_tokens > self.video_tokens
            || video_tokens % self.tokens_per_frame != 0
        {
            return Err(Error::Other(format!(
                "Fast-WAM video mask expects 1..={} whole frames of {} tokens, got {} tokens",
                self.video_grid[0], self.tokens_per_frame, video_tokens
            )));
        }
        let mut mask = AttentionMask::zeros(video_tokens, video_tokens)?;
        mask.fill_rect(0, self.tokens_per_frame, 0, self.tokens_per_frame);
        mask.fill_rect(self.tokens_per_frame, video_tokens, 0, video_tokens);
        Ok(mask)
    }

    /// Build the inference-time MoT mask for the selected official recipe.
    /// Video queries never read action keys. Action queries read one frame in
    /// Fast and the complete video prefix in IDM/Joint, plus all action keys.
    pub(crate) fn mot_attention_mask(&self, mode: FastWamMode) -> Result<AttentionMask> {
        let video_tokens = self.inference_video_tokens(mode);
        let total_tokens = video_tokens
            .checked_add(self.action_tokens)
            .ok_or_else(|| Error::Other("Fast-WAM MoT token count overflow".into()))?;
        let video_mask = self.video_attention_mask(video_tokens)?;
        let mut mask = AttentionMask::zeros(total_tokens, total_tokens)?;
        for row in 0..video_tokens {
            let source = &video_mask.values[row * video_tokens..(row + 1) * video_tokens];
            mask.values[row * total_tokens..row * total_tokens + video_tokens]
                .copy_from_slice(source);
        }
        let visible_video_tokens = match mode {
            FastWamMode::Fast => self.tokens_per_frame,
            FastWamMode::Idm | FastWamMode::Joint => video_tokens,
        };
        mask.fill_rect(video_tokens, total_tokens, 0, visible_video_tokens);
        mask.fill_rect(video_tokens, total_tokens, video_tokens, total_tokens);
        Ok(mask)
    }

    /// Per-video-token timestep values. The observation frame is always clean
    /// (`t=0`); every generated latent frame uses the current video timestep.
    pub(crate) fn video_token_timesteps(&self, mode: FastWamMode, timestep: f32) -> Vec<f32> {
        let tokens = self.inference_video_tokens(mode);
        let mut timesteps = vec![timestep; tokens];
        timesteps[..self.tokens_per_frame].fill(0.0);
        timesteps
    }

    pub(crate) fn action_rope(&self) -> Result<RopeTable> {
        rope_1d(self.head_dim, self.action_tokens)
    }

    pub(crate) fn video_rope(&self, mode: FastWamMode) -> Result<RopeTable> {
        let frames = match mode {
            FastWamMode::Fast => 1,
            FastWamMode::Idm | FastWamMode::Joint => self.video_grid[0],
        };
        rope_3d(
            self.head_dim,
            [frames, self.video_grid[1], self.video_grid[2]],
        )
    }
}

/// Fast-WAM's exact float64-generated 1D sinusoid, projected to f32 before the
/// caller converts it to the model's BF16 activation dtype. Rows are positions;
/// columns are all cosines followed by all sines.
pub(crate) fn sinusoidal_embedding_1d(dim: usize, positions: &[f32]) -> Result<Vec<f32>> {
    if dim == 0 || dim % 2 != 0 {
        return Err(Error::Other(format!(
            "Fast-WAM sinusoidal embedding dimension must be positive and even, got {dim}"
        )));
    }
    let half = dim / 2;
    let len = positions
        .len()
        .checked_mul(dim)
        .ok_or_else(|| Error::Other("Fast-WAM sinusoidal embedding size overflow".into()))?;
    let mut output = vec![0.0; len];
    for (row, &position) in positions.iter().enumerate() {
        let position = f64::from(position);
        for index in 0..half {
            let phase = position * 10_000.0f64.powf(-(index as f64) / half as f64);
            output[row * dim + index] = phase.cos() as f32;
            output[row * dim + half + index] = phase.sin() as f32;
        }
    }
    Ok(output)
}

fn rope_angles(dim: usize, position: usize, output: &mut Vec<(f32, f32)>) -> Result<()> {
    if dim == 0 || dim % 2 != 0 {
        return Err(Error::Other(format!(
            "Fast-WAM RoPE axis dimension must be positive and even, got {dim}"
        )));
    }
    for pair in 0..dim / 2 {
        let inverse_frequency = 1.0 / ROPE_THETA.powf((2 * pair) as f64 / dim as f64);
        let angle = position as f64 * inverse_frequency;
        output.push((angle.cos() as f32, angle.sin() as f32));
    }
    Ok(())
}

fn rope_1d(dim: usize, tokens: usize) -> Result<RopeTable> {
    let pairs_per_head = dim / 2;
    let capacity = tokens
        .checked_mul(pairs_per_head)
        .ok_or_else(|| Error::Other("Fast-WAM action RoPE size overflow".into()))?;
    let mut pairs = Vec::with_capacity(capacity);
    for position in 0..tokens {
        rope_angles(dim, position, &mut pairs)?;
    }
    Ok(split_rope_pairs(tokens, pairs_per_head, pairs))
}

fn rope_3d(head_dim: usize, grid: [usize; 3]) -> Result<RopeTable> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::Other(format!(
            "Fast-WAM RoPE head dimension must be positive and even, got {head_dim}"
        )));
    }
    // Official split: f gets the remainder, h and w each get floor(dim / 3).
    let spatial_dim = head_dim / 3;
    let axis_dims = [head_dim - 2 * spatial_dim, spatial_dim, spatial_dim];
    if axis_dims.iter().any(|dim| dim % 2 != 0) {
        return Err(Error::Other(format!(
            "Fast-WAM 3D RoPE axis dimensions must be even, got {axis_dims:?}"
        )));
    }
    let tokens = grid
        .into_iter()
        .try_fold(1usize, |product, axis| product.checked_mul(axis))
        .ok_or_else(|| Error::Other("Fast-WAM video RoPE size overflow".into()))?;
    let pairs_per_head = head_dim / 2;
    let capacity = tokens
        .checked_mul(pairs_per_head)
        .ok_or_else(|| Error::Other("Fast-WAM video RoPE table overflow".into()))?;
    let mut pairs = Vec::with_capacity(capacity);
    for frame in 0..grid[0] {
        for row in 0..grid[1] {
            for col in 0..grid[2] {
                rope_angles(axis_dims[0], frame, &mut pairs)?;
                rope_angles(axis_dims[1], row, &mut pairs)?;
                rope_angles(axis_dims[2], col, &mut pairs)?;
            }
        }
    }
    Ok(split_rope_pairs(tokens, pairs_per_head, pairs))
}

fn split_rope_pairs(tokens: usize, pairs_per_head: usize, pairs: Vec<(f32, f32)>) -> RopeTable {
    debug_assert_eq!(pairs.len(), tokens * pairs_per_head);
    let mut cos = Vec::with_capacity(pairs.len());
    let mut sin = Vec::with_capacity(pairs.len());
    for (cosine, sine) in pairs {
        cos.push(cosine);
        sin.push(sine);
    }
    RopeTable {
        tokens,
        pairs_per_head,
        cos,
        sin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(mode: FastWamMode) -> FastWamGeometry {
        FastWamGeometry::new(&FastWamConfig::libero_2cam224(mode)).unwrap()
    }

    #[test]
    fn release_geometry_matches_official_patch_order() {
        let geometry = geometry(FastWamMode::Fast);
        assert_eq!(geometry.video_grid, [3, 7, 14]);
        assert_eq!(geometry.tokens_per_frame, 98);
        assert_eq!(geometry.video_tokens, 294);
        assert_eq!(geometry.action_tokens, 32);
        assert_eq!(geometry.inference_video_tokens(FastWamMode::Fast), 98);
        assert_eq!(geometry.inference_video_tokens(FastWamMode::Idm), 294);
    }

    #[test]
    fn first_frame_causal_video_mask_matches_official_blocks() {
        let geometry = geometry(FastWamMode::Joint);
        let mask = geometry.video_attention_mask(294).unwrap();
        assert_eq!((mask.rows(), mask.cols()), (294, 294));
        assert!(mask.allows(0, 97));
        assert!(!mask.allows(0, 98));
        assert!(!mask.allows(97, 293));
        assert!(mask.allows(98, 0));
        assert!(mask.allows(98, 293));
        assert!(mask.allows(293, 293));
    }

    #[test]
    fn inference_mot_masks_distinguish_fast_from_idm_and_joint() {
        for (mode, video) in [
            (FastWamMode::Fast, 98),
            (FastWamMode::Idm, 294),
            (FastWamMode::Joint, 294),
        ] {
            let mask = geometry(mode).mot_attention_mask(mode).unwrap();
            assert_eq!((mask.rows(), mask.cols()), (video + 32, video + 32));
            for row in 0..mask.rows() {
                for col in 0..mask.cols() {
                    let allowed = row >= video || (col < video && (row >= 98 || col < 98));
                    assert_eq!(mask.allows(row, col), allowed, "{mode:?} [{row}, {col}]");
                }
            }
        }
    }

    #[test]
    fn video_token_timesteps_keep_observation_clean() {
        let geometry = geometry(FastWamMode::Joint);
        let values = geometry.video_token_timesteps(FastWamMode::Joint, 625.0);
        assert_eq!(values.len(), 294);
        assert!(values[..98].iter().all(|&value| value == 0.0));
        assert!(values[98..].iter().all(|&value| value == 625.0));
        assert_eq!(
            geometry.video_token_timesteps(FastWamMode::Fast, 625.0),
            vec![0.0; 98]
        );
    }

    #[test]
    fn sinusoidal_embedding_uses_cos_then_sin_and_float64_phase() {
        let values = sinusoidal_embedding_1d(8, &[0.0, 1.0]).unwrap();
        assert_eq!(&values[..4], &[1.0; 4]);
        assert_eq!(&values[4..8], &[0.0; 4]);
        let expected = [
            1.0f64.cos(),
            0.1f64.cos(),
            0.01f64.cos(),
            0.001f64.cos(),
            1.0f64.sin(),
            0.1f64.sin(),
            0.01f64.sin(),
            0.001f64.sin(),
        ];
        for (actual, expected) in values[8..].iter().zip(expected) {
            assert!((f64::from(*actual) - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn action_and_video_rope_follow_official_axis_split() {
        let geometry = geometry(FastWamMode::Joint);
        let action = geometry.action_rope().unwrap();
        assert_eq!((action.tokens(), action.pairs_per_head()), (32, 64));
        assert!(action.cos()[..64].iter().all(|&value| value == 1.0));
        assert!(action.sin()[..64].iter().all(|&value| value == 0.0));
        assert!((action.cos()[64] - 1.0f32.cos()).abs() < 1.0e-7);
        assert!((action.sin()[64] - 1.0f32.sin()).abs() < 1.0e-7);

        let video = geometry.video_rope(FastWamMode::Joint).unwrap();
        assert_eq!((video.tokens(), video.pairs_per_head()), (294, 64));
        // [44 frame dims, 42 height dims, 42 width dims] = [22, 21, 21] pairs.
        let width_first_pair = 22 + 21;
        assert!((video.cos()[64 + width_first_pair] - 1.0f32.cos()).abs() < 1.0e-7);
        assert!((video.sin()[64 + width_first_pair] - 1.0f32.sin()).abs() < 1.0e-7);
        let height_token = 14;
        assert!((video.cos()[height_token * 64 + 22] - 1.0f32.cos()).abs() < 1.0e-7);
        let next_frame_token = 98;
        assert!((video.cos()[next_frame_token * 64] - 1.0f32.cos()).abs() < 1.0e-7);
    }
}
