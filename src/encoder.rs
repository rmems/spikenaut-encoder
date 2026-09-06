use crate::time::TimeModel;
use crate::types::{EncodedOutput, SpikeEvent};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingEncoderConfigRepr"))]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingEncoderConfigRepr {
    v_th: f32,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingEncoderConfigRepr> for EmbeddingEncoderConfig {
    type Error = String;

    fn try_from(r: EmbeddingEncoderConfigRepr) -> Result<Self, String> {
        if r.v_th.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
            return Err("v_th must be positive".into());
        }
        Ok(Self { v_th: r.v_th })
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderState {
    pub membrane_potentials: Vec<f32>,
}

impl EncoderState {
    pub fn new_zeros(len: usize) -> Self {
        Self {
            membrane_potentials: vec![0.0; len],
        }
    }
}

/// Rate-style membrane encoder driven by a fixed embedding vector.
///
/// Accumulates normalized embedding components into per-channel membrane
/// potentials and emits a spike when the threshold is crossed (soft reset).
///
/// # No `SpikeSink` path
///
/// This type does not implement [`Encoder`](crate::Encoder) — it threads state
/// explicitly through [`forward`](Self::forward) rather than owning it — so it
/// is an explicit exception to the crate's allocation-reusing
/// [`SpikeSink`](crate::sink::SpikeSink) path. A sink would not make `forward`
/// allocation-free anyway: it returns a freshly built [`EncoderState`] per
/// call, which is the dominant allocation. Callers wanting a reusable spike
/// buffer can drain the returned `EncodedOutput::spikes` into one.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::encoder::{EmbeddingEncoderConfig, EmbeddingRateEncoder, EncoderState};
///
/// let enc = EmbeddingRateEncoder::new(&[0.5, 1.0], EmbeddingEncoderConfig { v_th: 0.4 });
/// let state = EncoderState::new_zeros(2);
/// let (out, next) = enc.forward(&state);
/// assert!(!out.spikes.is_empty());
/// assert_eq!(next.membrane_potentials.len(), 2);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingRateEncoderRepr"))]
pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingRateEncoderRepr {
    config: EmbeddingEncoderConfig,
    normalized_embeddings: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingRateEncoderRepr> for EmbeddingRateEncoder {
    type Error = String;

    fn try_from(r: EmbeddingRateEncoderRepr) -> Result<Self, String> {
        if r.normalized_embeddings.iter().any(|v| !v.is_finite()) {
            return Err("normalized_embeddings must be finite".into());
        }
        if r.normalized_embeddings.len() > u16::MAX as usize + 1 {
            return Err("too many channels (max 65536)".into());
        }
        Ok(Self {
            config: r.config,
            normalized_embeddings: r.normalized_embeddings,
        })
    }
}

impl EmbeddingRateEncoder {
    /// Time model for spikes emitted by [`forward`](Self::forward).
    ///
    /// One `forward` call is one tick, and every spike lands at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO): this encoder reports
    /// *whether* a channel crossed threshold in the step, not *when* within it.
    /// Ticks are dimensionless — `EmbeddingEncoderConfig` carries no `dt`, so
    /// the caller owns the physical step duration.
    #[inline]
    pub const fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    pub fn new(embeddings: &[f32], config: EmbeddingEncoderConfig) -> Self {
        if config.v_th.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
            panic!("v_th must be positive");
        }
        assert!(
            embeddings.len() <= u16::MAX as usize + 1,
            "too many channels (max 65536)"
        );

        let min_val = embeddings.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let epsilon = 1e-5f32;
        let safe_range = range + epsilon;

        let normalized: Vec<f32> = embeddings
            .iter()
            .map(|&x| (x - min_val) / safe_range)
            .collect();

        Self {
            config,
            normalized_embeddings: normalized,
        }
    }

    pub fn forward(&self, prev_state: &EncoderState) -> (EncodedOutput, EncoderState) {
        let mut new_potentials = prev_state.membrane_potentials.clone();
        let mut output = EncodedOutput::new();

        for (i, (pot, &emb)) in new_potentials
            .iter_mut()
            .zip(self.normalized_embeddings.iter())
            .enumerate()
        {
            *pot += emb;

            if *pot >= self.config.v_th {
                output.spikes.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    true,
                ));
                *pot -= self.config.v_th; // Soft reset
            }
        }

        (
            output,
            EncoderState {
                membrane_potentials: new_potentials,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_rate_encoder_basic() {
        let config = EmbeddingEncoderConfig { v_th: 0.9 };
        let embeddings = [0.5, 1.0, 0.0];
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);

        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 1);

        let (output2, _) = encoder.forward(&next_state);
        // Channel 0: 0.5 + 0.5 = 1.0 > 0.9 -> spike
        // Channel 1: (1.0-0.9) + 1.0 = 1.1 > 0.9 -> spike
        assert_eq!(output2.spikes.len(), 2);
    }

    #[test]
    #[should_panic(expected = "v_th must be positive")]
    fn test_embedding_encoder_config_invalid_vth() {
        let _ = EmbeddingRateEncoder::new(&[0.5], EmbeddingEncoderConfig { v_th: 0.0 });
    }

    #[test]
    fn test_encoder_state_new_zeros() {
        let state = EncoderState::new_zeros(5);
        assert_eq!(state.membrane_potentials.len(), 5);
        assert!(state.membrane_potentials.iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_embedding_rate_encoder_deserialize_rejects_too_many_channels() {
        let embeddings: Vec<f32> = vec![0.0; (u16::MAX as usize) + 2];
        let value = serde_json::json!({
            "config": {"v_th": 1.0},
            "normalized_embeddings": embeddings,
        });
        let res: Result<EmbeddingRateEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }
}

#[cfg(test)]
mod forward_coverage_tests {
    use super::*;

    #[test]
    fn embedding_rate_encoder_initializes_normalized_values() {
        let embeddings = vec![1.0, 2.0, 3.0, 5.0];
        let encoder = EmbeddingRateEncoder::new(&embeddings, EmbeddingEncoderConfig { v_th: 1.0 });
        assert_eq!(encoder.normalized_embeddings.len(), 4);
        assert!((encoder.normalized_embeddings[0] - 0.0).abs() < 1e-5);
        assert!((encoder.normalized_embeddings[3] - (4.0 / 4.00001)).abs() < 1e-5);
    }

    #[test]
    fn embedding_rate_encoder_forward_without_spikes() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let encoder = EmbeddingRateEncoder::new(&embeddings, EmbeddingEncoderConfig { v_th: 10.0 });
        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);

        assert!(output.spikes.is_empty());
        assert_eq!(next_state.membrane_potentials.len(), 3);
        assert_eq!(
            next_state.membrane_potentials,
            encoder.normalized_embeddings
        );
    }

    #[test]
    fn embedding_rate_encoder_forward_soft_resets_spikes() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let encoder = EmbeddingRateEncoder::new(&embeddings, EmbeddingEncoderConfig { v_th: 0.4 });
        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);

        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 1);
        assert_eq!(output.spikes[1].channel, 2);
        assert!((next_state.membrane_potentials[0] - 0.0).abs() < 1e-5);
        assert!(
            (next_state.membrane_potentials[1] - (encoder.normalized_embeddings[1] - 0.4)).abs()
                < 1e-5
        );
        assert!(
            (next_state.membrane_potentials[2] - (encoder.normalized_embeddings[2] - 0.4)).abs()
                < 1e-5
        );
    }

    #[test]
    fn embedding_rate_encoder_accumulates_across_steps() {
        let embeddings = vec![1.0, 3.0];
        let encoder = EmbeddingRateEncoder::new(&embeddings, EmbeddingEncoderConfig { v_th: 0.6 });
        let mut state = EncoderState::new_zeros(2);

        for _ in 0..3 {
            let (output, next_state) = encoder.forward(&state);
            assert_eq!(output.spikes.len(), 1);
            assert_eq!(output.spikes[0].channel, 1);
            state = next_state;
        }

        assert!((state.membrane_potentials[0] - 0.0).abs() < 1e-5);
        assert!(
            (state.membrane_potentials[1] - (3.0 * encoder.normalized_embeddings[1] - 1.8)).abs()
                < 1e-5
        );
    }

    #[test]
    fn embedding_rate_encoder_handles_equal_embeddings() {
        let embeddings = vec![2.5, 2.5, 2.5];
        let encoder = EmbeddingRateEncoder::new(&embeddings, EmbeddingEncoderConfig { v_th: 0.5 });
        assert!(
            encoder
                .normalized_embeddings
                .iter()
                .all(|value| *value == 0.0)
        );

        let (output, next_state) = encoder.forward(&EncoderState::new_zeros(3));
        assert!(output.spikes.is_empty());
        assert_eq!(next_state.membrane_potentials, vec![0.0, 0.0, 0.0]);
    }
}
