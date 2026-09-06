use crate::prelude::*;

/// Encodes based on the rate of change (derivative) of the input.
///
/// Fires an excitatory spike when the positive change exceeds a threshold,
/// and an inhibitory spike when the negative change exceeds the threshold.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = DerivativeEncoder::try_new(vec![0.2])?;
/// let _ = enc.encode_step(&[0.0]); // seed previous sample
/// let out = enc.encode_step(&[0.5]); // +0.5 change exceeds threshold
/// assert_eq!(out.spikes.len(), 1);
/// assert!(out.spikes[0].polarity);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "DerivativeEncoderRepr"))]
pub struct DerivativeEncoder {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct DerivativeEncoderRepr {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<DerivativeEncoderRepr> for DerivativeEncoder {
    type Error = String;

    fn try_from(r: DerivativeEncoderRepr) -> Result<Self, String> {
        if r.last_values.len() != r.thresholds.len() {
            return Err(format!(
                "mismatched last_values length ({}) and thresholds length ({})",
                r.last_values.len(),
                r.thresholds.len()
            ));
        }
        if r.last_values.iter().any(|v| !v.is_finite()) {
            return Err("last_values must be finite".into());
        }
        let mut encoder = Self::try_new(r.thresholds).map_err(|error| error.to_string())?;
        encoder.last_values = r.last_values;
        Ok(encoder)
    }
}

impl DerivativeEncoder {
    /// Creates a new `DerivativeEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if any threshold is non-finite/negative or the channel count is too large.
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self::try_new(thresholds).expect("invalid DerivativeEncoder configuration")
    }

    /// Creates a new `DerivativeEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// Each threshold must be finite and non-negative; channel count must fit `u16` IDs.
    pub fn try_new(thresholds: Vec<f32>) -> Result<Self, EncoderError> {
        crate::error::validate_channel_count(thresholds.len())?;
        for &threshold in &thresholds {
            crate::error::validate_non_negative_finite("threshold", threshold)?;
        }
        let num_channels = thresholds.len();
        Ok(Self {
            last_values: vec![0.0; num_channels],
            thresholds,
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart.
    fn encode_step_into_sink<S: SpikeSink + ?Sized>(
        &mut self,
        current_values: &[f32],
        sink: &mut S,
    ) {
        for (i, &current_val) in current_values.iter().enumerate() {
            if i >= self.thresholds.len() {
                break;
            }

            let delta = current_val - self.last_values[i];

            // Excitatory spike on positive jump exceeding threshold
            if delta > self.thresholds[i] {
                sink.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    true,
                ));
            }
            // Inhibitory/Negative spike on sudden drop
            else if delta < -self.thresholds[i] {
                sink.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    false,
                ));
            }

            self.last_values[i] = current_val;
        }
    }
}

impl Encoder for DerivativeEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step(input)
    }

    fn encode_step(&mut self, current_values: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_step_into_sink(current_values, &mut output.spikes);
        output
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(input, sink));
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_into(input, sink);
    }

    /// One call is one tick; `encode` and `encode_step` are the same path.
    ///
    /// A channel emits at most one spike per call — excitatory on a jump,
    /// inhibitory on a drop — always at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO). Ticks are
    /// dimensionless: the derivative is per *step*, so the caller owns the
    /// physical step duration.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_encoder_basic() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // Initial jump
        let output = encoder.encode(&[1.5, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);

        // Stay same
        let output = encoder.encode(&[1.5, 1.5]);
        assert!(output.spikes.is_empty());

        // Jump down
        let output = encoder.encode(&[0.0, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(!output.spikes[0].polarity);

        // Jump up on channel 1
        let output = encoder.encode(&[0.0, 4.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 1);
        assert!(output.spikes[0].polarity);
    }

    #[test]
    fn test_derivative_encoder_reset() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        encoder.encode(&[5.0]);
        encoder.reset();
        assert_eq!(encoder.last_values[0], 0.0);
    }

    #[test]
    fn test_derivative_encoder_empty_and_mismatched() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[2.0, 3.0]);
        assert_eq!(output.spikes.len(), 1); // Only channel 0 should be processed
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_derivative_serde_rejects_too_many_channels() {
        let values: Vec<f32> = vec![0.0; (u16::MAX as usize) + 2];
        let value = serde_json::json!({
            "last_values": values.clone(),
            "thresholds": values,
        });
        let res: Result<DerivativeEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }

    #[test]
    fn test_derivative_encoder_try_new_validation() {
        assert!(DerivativeEncoder::try_new(vec![0.0, 1.0]).is_ok());
        assert_eq!(
            DerivativeEncoder::try_new(vec![f32::NAN]).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DerivativeEncoder::try_new(vec![-1.0]).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DerivativeEncoder::try_new(vec![1.0; u16::MAX as usize + 2]).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
    }
}

#[cfg(test)]
mod branch_coverage_tests {
    use super::*;

    #[test]
    fn derivative_encoder_initializes_channel_state() {
        let encoder = DerivativeEncoder::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.thresholds, vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.last_values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn derivative_encoder_tracks_positive_and_negative_steps() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        let output = encoder.encode_step(&[1.5, -2.5]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);
        assert_eq!(output.spikes[1].channel, 1);
        assert!(!output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![1.5, -2.5]);

        let output = encoder.encode_step(&[2.0, -1.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![2.0, -1.0]);

        let output = encoder.encode_step(&[0.0, 2.0]);
        assert_eq!(output.spikes.len(), 2);
        assert!(!output.spikes[0].polarity);
        assert!(output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![0.0, 2.0]);
    }

    #[test]
    fn derivative_encoder_does_not_fire_at_threshold() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[1.0, -2.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![1.0, -2.0]);
    }

    #[test]
    fn derivative_encoder_handles_channel_count_mismatches() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0, -3.0, 5.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(encoder.last_values, vec![2.0, -3.0]);

        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(encoder.last_values, vec![2.0, 0.0]);
    }
}
