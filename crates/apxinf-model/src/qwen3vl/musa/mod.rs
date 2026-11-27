//! Native MUSA Qwen3-VL executor boundary.

mod executor;
mod kv_cache;
mod language;
mod vision;
pub use vision::NativeVisionOutput;
pub mod load;
pub mod runtime;
pub mod weights;
pub use executor::Qwen3VlMusaExecutor;

#[derive(Debug, Default)]
pub struct NativeSequenceState {
    position: usize,
}
impl NativeSequenceState {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        self.position = 0;
    }
}
pub fn validate_decode_transition(
    state: &mut NativeSequenceState,
    tokens: usize,
    start_position: usize,
) -> Result<(), String> {
    if tokens == 0 {
        return Err("native Qwen3-VL transition requires tokens".into());
    }
    if start_position != state.position {
        return Err(format!(
            "decode position gap: expected {}, got {}",
            state.position, start_position
        ));
    }
    state.position = state
        .position
        .checked_add(tokens)
        .ok_or_else(|| "position overflow".to_string())?;
    Ok(())
}
