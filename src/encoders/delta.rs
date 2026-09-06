use crate::prelude::*;

/// A simple delta-based encoder.
///
/// Fires a spike when the absolute difference between the current input and the last
/// encoded value exceeds a threshold. This is useful for event-based encoding where
/// only changes in the input signal are relevant.
///
/// # Mathematical Model
///
/// ```text
/// delta = |current_value - last_value|
/// spike if delta > threshold
/// ```
///
/// # When to Use
///
/// - Event-based encoding where changes are more important than absolute values
/// - Sensor data where baseline can drift but changes are meaningful
/// - Reducing power consumption by only encoding when changes occur
///
/// # Parameters
///
/// - `threshold`: Minimum change required to trigger a spike
/// - `num_channels`: Number of input channels to track
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = DeltaEncoder::try_new(0.1, 2)?;
/// // Baseline starts at zeros; last_values update only when a spike fires.
/// // A jump of 0.5 from 0 exceeds threshold 0.1 on channel 0.
/// let out = enc.encode(&[0.5, 0.0]);
/// assert!(!out.spikes.is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DeltaEncoder {
    last_values: Vec<f32>,
    threshold: f32,
}

impl DeltaEncoder {
    /// Creates a new `DeltaEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    pub fn new(threshold: f32, num_channels: usize) -> Self {
        Self::try_new(threshold, num_channels).expect("invalid DeltaEncoder configuration")
    }

    /// Creates a new `DeltaEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// `threshold == 0.0` is valid and means any nonzero change fires a spike
    /// (`delta > 0`).
    pub fn try_new(threshold: f32, num_channels: usize) -> Result<Self, EncoderError> {
        crate::error::validate_non_negative_finite("threshold", threshold)?;
        crate::error::validate_channel_count(num_channels)?;
        Ok(Self {
            last_values: vec![0.0; num_channels],
            threshold,
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart.
    fn encode_with_threshold_scale_into<S: SpikeSink + ?Sized>(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
        sink: &mut S,
    ) {
        let effective_threshold = (self.threshold * threshold_scale).max(0.0);

        for (i, &value) in input.iter().enumerate() {
            if i >= self.last_values.len() {
                break;
            }
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let delta = (value - self.last_values[i]).abs();
            if delta > effective_threshold {
                sink.push(SpikeEvent::at_step_start(
                    channel,
                    value > self.last_values[i],
                ));
                self.last_values[i] = value;
            }
        }
    }

    fn encode_with_threshold_scale(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_with_threshold_scale_into(input, threshold_scale, &mut output.spikes);
        output
    }

    /// Streaming inputs are truncated to the configured channel count.
    fn clamp_to_channels<'a>(&self, input: &'a [f32]) -> &'a [f32] {
        if input.len() > self.last_values.len() {
            &input[..self.last_values.len()]
        } else {
            input
        }
    }

    /// Encodes input using neuromodulator-driven gain curves.
    ///
    /// Inherent wrapper so callers need not import [`ModulatedEncoder`].
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_with_modulators(self, input, modulators, gain_curves)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_step_with_modulators(
            self,
            input,
            modulators,
            gain_curves,
        )
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DeltaEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            last_values: Vec<f32>,
            threshold: f32,
        }
        let helper = Helper::deserialize(deserializer)?;
        let mut encoder = Self::try_new(helper.threshold, helper.last_values.len())
            .map_err(serde::de::Error::custom)?;
        if helper.last_values.iter().any(|value| !value.is_finite()) {
            return Err(serde::de::Error::custom("last_values must be finite"));
        }
        encoder.last_values = helper.last_values;
        Ok(encoder)
    }
}

impl Encoder for DeltaEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode(safe_input)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(input, 1.0, sink)
        });
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        let safe_input = self.clamp_to_channels(input);
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(safe_input, 1.0, sink)
        });
    }

    /// One call is one tick; batch and streaming are identical here.
    ///
    /// A channel emits at most one spike per call, always at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO): a threshold crossing
    /// is attributed to the step it was observed in, not to a sub-step instant.
    /// Ticks are dimensionless — `DeltaEncoder` has no sampling interval, so the
    /// caller owns the physical step duration.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

impl ModulatedEncoder for DeltaEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_threshold_scale(input, gains.sanitize().threshold_scale)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let threshold_scale = gains.sanitize().threshold_scale;
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(input, threshold_scale, sink)
        });
    }
}

/// Simplified: delta-based spike generation (per feature).
///
/// This is a utility function that takes a slice of deltas and returns a boolean spike train.
/// It can be used to feed the resulting binary/event sequences into LIF/RSNN layers.
pub fn encode_deltas_to_spikes(deltas: &[f32], threshold: f32) -> Vec<bool> {
    deltas.iter().map(|&d| d.abs() > threshold).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new(2.0, 1);
        let output = encoder.encode(&[1.0]); // 1.0 - 0.0 = 1.0 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[3.5]); // 3.5 - 0.0 = 3.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert!(output.spikes[0].polarity);

        let output = encoder.encode(&[4.0]); // 4.0 - 3.5 = 0.5 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[1.0]); // 1.0 - 3.5 = -2.5.abs() = 2.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert!(!output.spikes[0].polarity);
    }

    #[test]
    fn test_delta_encoder_encode_step() {
        let mut encoder = DeltaEncoder::new(2.0, 2);
        let output = encoder.encode_step(&[3.0, 3.0, 3.0]); // 3rd channel ignored
        assert_eq!(output.spikes.len(), 2);
    }

    #[test]
    fn test_delta_encoder_multi_channel_reset() {
        let mut encoder = DeltaEncoder::new(1.0, 2);
        encoder.encode(&[2.0, 2.0]);
        assert_eq!(encoder.last_values, vec![2.0, 2.0]);
        encoder.reset();
        assert_eq!(encoder.last_values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_delta_encoder_empty_input() {
        let mut encoder = DeltaEncoder::new(1.0, 5);
        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_encode_deltas_to_spikes() {
        let deltas = [0.1, 0.5, -0.8, 1.2];
        let threshold = 0.7;
        let spikes = encode_deltas_to_spikes(&deltas, threshold);
        assert_eq!(spikes, vec![false, false, true, true]);
    }

    #[test]
    fn test_delta_encoder_modulators_reduce_threshold() {
        let mut encoder = DeltaEncoder::new(1.0, 1);
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(encoder.encode(&[0.75]).spikes.is_empty());
        encoder.reset();

        let modulated = encoder.encode_with_modulators(&[0.75], &modulators, &gain_curves);
        assert_eq!(modulated.spikes.len(), 1);
    }

    #[test]
    fn test_delta_encoder_encode_step_with_modulators() {
        let mut encoder = DeltaEncoder::new(1.0, 1);
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(
            encoder
                .encode_step_with_modulators(&[0.0], &modulators, &gain_curves)
                .spikes
                .is_empty()
        );
        let modulated = encoder.encode_step_with_modulators(&[0.75], &modulators, &gain_curves);
        assert_eq!(modulated.spikes.len(), 1);
    }

    #[test]
    fn test_delta_encoder_step_shorter_input() {
        let mut encoder = DeltaEncoder::new(1.0, 2);
        let output = encoder.encode_step(&[2.0]);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_delta_encoder_truncates_excess_channels() {
        let mut encoder = DeltaEncoder::new(1.0, 1);
        let output = encoder.encode(&[2.0, 3.0]);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_delta_encoder_zero_threshold_scale_spikes_on_any_change() {
        let mut encoder = DeltaEncoder::new(1.0, 1);
        encoder.encode(&[0.0]);
        let output = encoder.encode_with_threshold_scale(&[0.01], 0.0);
        assert_eq!(output.spikes.len(), 1);
    }
    #[test]
    fn test_delta_encoder_zero_threshold_spikes_on_any_change() {
        let mut encoder = DeltaEncoder::new(0.0, 1);
        encoder.encode(&[0.0]);
        let output = encoder.encode(&[0.01]);
        assert_eq!(output.spikes.len(), 1);
        let quiet = encoder.encode(&[0.01]);
        assert!(quiet.spikes.is_empty());
    }

    #[test]
    fn test_delta_encoder_try_new_validation() {
        assert!(DeltaEncoder::try_new(0.0, 1).is_ok());
        assert_eq!(
            DeltaEncoder::try_new(-1.0, 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DeltaEncoder::try_new(f32::NAN, 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DeltaEncoder::try_new(1.0, u16::MAX as usize + 2).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
    }
}
