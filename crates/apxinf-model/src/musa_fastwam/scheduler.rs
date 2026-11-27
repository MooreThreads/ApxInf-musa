use apxinf_core::{Error, Result};

/// Host-built constants for the official continuous flow-matching schedule.
///
/// Building twenty scalar timestep/delta values is request-independent. The
/// model uploads them once and keeps all per-step tensor updates on MUSA.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FlowMatchSchedule {
    timesteps: Vec<f32>,
    deltas: Vec<f32>,
}

impl FlowMatchSchedule {
    pub(crate) fn new(num_steps: usize, num_train_timesteps: usize, shift: f32) -> Result<Self> {
        if num_steps == 0 || num_train_timesteps == 0 {
            return Err(Error::Other(
                "Fast-WAM flow schedule step counts must be positive".into(),
            ));
        }
        if !shift.is_finite() || shift <= 0.0 {
            return Err(Error::Other(
                "Fast-WAM flow schedule shift must be finite and positive".into(),
            ));
        }
        let sigma: Vec<f32> = (0..=num_steps)
            .map(|index| {
                let u = 1.0 - index as f32 / num_steps as f32;
                shift * u / (1.0 + (shift - 1.0) * u)
            })
            .collect();
        let timesteps = sigma[..num_steps]
            .iter()
            .map(|value| value * num_train_timesteps as f32)
            .collect();
        let deltas = sigma.windows(2).map(|pair| pair[1] - pair[0]).collect();
        Ok(Self { timesteps, deltas })
    }

    pub(crate) fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub(crate) fn deltas(&self) -> &[f32] {
        &self.deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_one_is_uniform_from_one_to_zero() {
        let schedule = FlowMatchSchedule::new(4, 1000, 1.0).unwrap();
        assert_eq!(schedule.timesteps(), &[1000.0, 750.0, 500.0, 250.0]);
        assert_eq!(schedule.deltas(), &[-0.25, -0.25, -0.25, -0.25]);
    }

    #[test]
    fn shifted_schedule_matches_official_phi() {
        let schedule = FlowMatchSchedule::new(2, 1000, 5.0).unwrap();
        let expected_timesteps = [1000.0, 2500.0 / 3.0];
        let expected_deltas = [-1.0 / 6.0, -5.0 / 6.0];
        for (actual, expected) in schedule.timesteps().iter().zip(expected_timesteps) {
            assert!((actual - expected).abs() <= 1.0e-4);
        }
        for (actual, expected) in schedule.deltas().iter().zip(expected_deltas) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn invalid_schedule_parameters_fail_closed() {
        assert!(FlowMatchSchedule::new(0, 1000, 1.0).is_err());
        assert!(FlowMatchSchedule::new(20, 0, 1.0).is_err());
        assert!(FlowMatchSchedule::new(20, 1000, 0.0).is_err());
    }
}
