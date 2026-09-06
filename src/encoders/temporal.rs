use crate::prelude::*;
use std::collections::VecDeque;

/// Encodes temporal patterns by tracking history of values per channel.
///
/// Fires a spike when the rate of change exceeds configurable thresholds.
/// Useful for detecting sudden changes or motion in sensor signals.
///
/// # Mathematical Model
///
/// Computes the difference between recent average (last 3 values) and older average
/// (previous 3 values before that). A spike is generated when this change exceeds
/// the threshold:
///
/// ```text
/// change = |mean(history[-3:]) - mean(history[-6:-3])|
/// spike if change > threshold
/// ```
///
/// # When to Use
///
/// - Detecting sudden changes in signal (edge detection)
/// - Motion detection in video or sensor streams
/// - Event-based encoding where changes are more important than absolute values
///
/// # Parameters
///
/// - `history_depth`: How many past values to track per channel
/// - `change_thresholds`: Vec of (threshold, spike_value) pairs - fires when change exceeds threshold
/// - `num_channels`: Number of input channels
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// // history_depth must be at least 6 for the dual-window change detector.
/// let mut enc = TemporalEncoder::try_new(6, vec![(0.5, 1)], 1)?;
/// for v in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0] {
///     let _ = enc.encode_step(&[v]);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemporalEncoder {
    history: Vec<VecDeque<f32>>,
    history_depth: usize,
    change_thresholds: Vec<(f32, u16)>,
}

impl TemporalEncoder {
    /// Creates a new `TemporalEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if `history_depth < 6` or `num_channels` is unsupported.
    pub fn new(
        history_depth: usize,
        change_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Self {
        Self::try_new(history_depth, change_thresholds, num_channels)
            .expect("invalid TemporalEncoder configuration")
    }

    /// Creates a new `TemporalEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// Each threshold in `change_thresholds` must be finite and non-negative.
    pub fn try_new(
        history_depth: usize,
        change_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Result<Self, EncoderError> {
        if history_depth < 6 {
            return Err(EncoderError::HistoryDepthTooSmall { minimum: 6 });
        }
        for &(threshold, _) in &change_thresholds {
            crate::error::validate_non_negative_finite("change_threshold", threshold)?;
        }
        crate::error::validate_channel_count(num_channels)?;
        Ok(Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            history_depth,
            change_thresholds,
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
        for (i, &value) in input.iter().enumerate() {
            if i >= self.history.len() {
                break;
            }
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let channel_history = &mut self.history[i];
            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            if channel_history.len() < 6 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(3).sum::<f32>() / 3.0;
            let older_avg = channel_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
            let change = (recent_avg - older_avg).abs();

            for &(threshold, _spike_val) in self.change_thresholds.iter().rev() {
                if change > (threshold * threshold_scale).max(0.0) {
                    // Or use spike_val to determine polarity/strength.
                    sink.push(SpikeEvent::at_step_start(channel, true));
                    break; // Only fire one spike per channel per step
                }
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
        if input.len() > self.history.len() {
            &input[..self.history.len()]
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

impl Encoder for TemporalEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale(safe_input, 1.0)
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
    /// A channel emits at most one spike per call, at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO) — the pattern lives
    /// in the *sequence* of steps, not in offsets within one. Ticks are
    /// dimensionless: one tick is one sample of the history window.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
    }
}

impl ModulatedEncoder for TemporalEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale(safe_input, gains.sanitize().threshold_scale)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let safe_input = self.clamp_to_channels(input);
        let threshold_scale = gains.sanitize().threshold_scale;
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(safe_input, threshold_scale, sink)
        });
    }

    /// Mirrors [`encode_step_with_gains`], which this encoder leaves at its
    /// default of [`encode_with_gains`]. Without this the trait default would
    /// build and drain an intermediate `EncodedOutput`, so the streaming
    /// modulated path would allocate on every step.
    ///
    /// [`encode_step_with_gains`]: ModulatedEncoder::encode_step_with_gains
    /// [`encode_with_gains`]: ModulatedEncoder::encode_with_gains
    fn encode_step_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gains, sink);
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TemporalEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::collections::VecDeque;

        #[derive(serde::Deserialize)]
        struct Helper {
            history: Vec<VecDeque<f32>>,
            history_depth: usize,
            change_thresholds: Vec<(f32, u16)>,
        }

        let helper = Helper::deserialize(deserializer)?;

        if helper.history_depth < 6 {
            return Err(serde::de::Error::custom(
                EncoderError::HistoryDepthTooSmall { minimum: 6 },
            ));
        }
        crate::error::validate_channel_count(helper.history.len())
            .map_err(serde::de::Error::custom)?;

        // Match try_new: reject non-finite / negative thresholds on load.
        for &(threshold, _) in &helper.change_thresholds {
            crate::error::validate_non_negative_finite("change_threshold", threshold)
                .map_err(serde::de::Error::custom)?;
        }

        for (i, deque) in helper.history.iter().enumerate() {
            if deque.len() > helper.history_depth {
                return Err(serde::de::Error::custom(format!(
                    "history channel {} length ({}) exceeds history_depth ({})",
                    i,
                    deque.len(),
                    helper.history_depth
                )));
            }
        }

        Ok(Self {
            history: helper.history,
            history_depth: helper.history_depth,
            change_thresholds: helper.change_thresholds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_encoder() {
        let mut encoder = TemporalEncoder::new(6, vec![(2.0, 1), (5.0, 2)], 1);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[8.0]);
        let _output = encoder.encode(&[8.0]);
        let output = encoder.encode(&[8.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_temporal_encoder_modulators_reduce_threshold() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode(&[1.0]);
        }
        for _ in 0..2 {
            encoder.encode(&[5.0]);
        }
        assert!(encoder.encode(&[5.0]).spikes.is_empty());

        encoder.reset();

        for _ in 0..3 {
            encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_temporal_encoder_encode_with_modulators() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_temporal_encoder_step_longer_input() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 2);
        let output = encoder.encode_step(&[1.0, 2.0, 3.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_history_channel_too_long() {
        let json = r#"{
            "history": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "history_depth": 6,
            "change_thresholds": []
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_rejects_invalid_thresholds() {
        let negative = r#"{
            "history": [[]],
            "history_depth": 6,
            "change_thresholds": [[-0.5, 1]]
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(negative);
        assert!(
            res.is_err(),
            "negative change_threshold must fail deserialize"
        );

        let ok = r#"{
            "history": [[]],
            "history_depth": 6,
            "change_thresholds": [[0.0, 1], [1.5, 2]]
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(ok);
        assert!(res.is_ok());
    }
    #[test]
    fn test_temporal_encoder_try_new_validation() {
        assert_eq!(
            TemporalEncoder::try_new(5, vec![(1.0, 1)], 1).err(),
            Some(EncoderError::HistoryDepthTooSmall { minimum: 6 })
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![(1.0, 1)], u16::MAX as usize + 2).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![(f32::NAN, 1)], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "change_threshold"
            })
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![(-0.5, 1)], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "change_threshold"
            })
        );
        assert!(TemporalEncoder::try_new(6, vec![(0.0, 1)], 1).is_ok());
    }
}
