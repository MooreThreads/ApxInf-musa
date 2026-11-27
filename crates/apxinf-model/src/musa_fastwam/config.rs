use apxinf_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Which official Fast-WAM inference recipe is executed.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FastWamMode {
    /// Encode the observed first frame, prefill video K/V once, denoise action.
    Fast,
    /// Denoise future video first, then infer action from final video K/V.
    Idm,
    /// Jointly denoise video and action, then decode generated video.
    Joint,
}

impl FromStr for FastWamMode {
    type Err = ();

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "fast" => Ok(Self::Fast),
            "idm" => Ok(Self::Idm),
            "joint" => Ok(Self::Joint),
            _ => Err(()),
        }
    }
}

/// Fixed first-release profile for the official LIBERO two-camera checkpoint.
///
/// These values come from the official `fastwam.yaml` and `libero_2cam.yaml`
/// at commit `7faa71108368fbb3b6885649f112af607427a2d4`. Keeping the
/// admitted profile explicit prevents a nominally configurable shape from
/// reaching a weight or mask layout that has not been reference-checked.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FastWamConfig {
    pub mode: FastWamMode,
    pub batch_size: usize,
    pub num_views: usize,
    pub camera_height: usize,
    pub camera_width: usize,
    pub merged_height: usize,
    pub merged_width: usize,
    pub context_len: usize,
    pub text_dim: usize,
    pub proprio_dim: usize,
    pub action_horizon: usize,
    pub action_dim: usize,
    pub num_video_frames: usize,
    pub latent_channels: usize,
    pub latent_temporal: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub patch_size: [usize; 3],
    pub video_hidden_dim: usize,
    pub video_ffn_dim: usize,
    pub action_hidden_dim: usize,
    pub action_ffn_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub freq_dim: usize,
    pub norm_eps: f32,
    pub num_inference_steps: usize,
    pub num_train_timesteps: usize,
    pub video_shift: f32,
    pub action_shift: f32,
}

impl FastWamConfig {
    pub fn libero_2cam224(mode: FastWamMode) -> Self {
        Self {
            mode,
            batch_size: 1,
            num_views: 2,
            camera_height: 224,
            camera_width: 224,
            merged_height: 224,
            merged_width: 448,
            context_len: 128,
            text_dim: 4096,
            proprio_dim: 8,
            action_horizon: 32,
            action_dim: 7,
            num_video_frames: 9,
            latent_channels: 48,
            latent_temporal: 3,
            latent_height: 14,
            latent_width: 28,
            patch_size: [1, 2, 2],
            video_hidden_dim: 3072,
            video_ffn_dim: 14336,
            action_hidden_dim: 1024,
            action_ffn_dim: 4096,
            num_heads: 24,
            head_dim: 128,
            num_layers: 30,
            freq_dim: 256,
            norm_eps: 1.0e-6,
            num_inference_steps: 20,
            num_train_timesteps: 1000,
            video_shift: 5.0,
            action_shift: 5.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let expected = Self::libero_2cam224(self.mode);
        if self != &expected {
            return Err(Error::Other(format!(
                "MUSA Fast-WAM first release admits only the fixed LIBERO two-camera 224 profile; got {self:?}"
            )));
        }
        if self.video_hidden_dim != self.num_heads * self.head_dim {
            return Err(Error::Other(
                "MUSA Fast-WAM video hidden width must equal heads * head_dim".into(),
            ));
        }
        if self.num_video_frames % 4 != 1 {
            return Err(Error::Other(
                "MUSA Fast-WAM video frame count must satisfy T % 4 == 1".into(),
            ));
        }
        Ok(())
    }

    pub const fn action_shape(&self) -> [usize; 2] {
        [self.action_horizon, self.action_dim]
    }

    pub const fn video_latent_shape(&self) -> [usize; 5] {
        [
            self.batch_size,
            self.latent_channels,
            self.latent_temporal,
            self.latent_height,
            self.latent_width,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_profiles_are_valid_for_all_three_modes() {
        for mode in [FastWamMode::Fast, FastWamMode::Idm, FastWamMode::Joint] {
            let config = FastWamConfig::libero_2cam224(mode);
            config.validate().unwrap();
            assert_eq!((config.action_shift, config.video_shift), (5.0, 5.0));
            assert_eq!(config.action_shape(), [32, 7]);
            assert_eq!(config.video_latent_shape(), [1, 48, 3, 14, 28]);
        }
    }

    #[test]
    fn changed_profile_is_rejected_until_it_has_reference_evidence() {
        let mut config = FastWamConfig::libero_2cam224(FastWamMode::Fast);
        config.num_views = 3;
        assert!(config.validate().is_err());
    }
}
