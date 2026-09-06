use crate::prelude::*;
use std::collections::VecDeque;
use std::fmt;

/// Errors that can occur when initializing a [`PredictiveEncoder`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveEncoderError {
    /// `history_depth` was less than 5 (the minimum window used by the predictor).
    HistoryDepthTooSmall,
    /// `num_channels` exceeds the `u16` channel-ID range used when emitting spikes.
    ///
    /// Valid channel indices are `0..=u16::MAX`, so at most `u16::MAX as usize + 1` channels.
    NumChannelsTooLarge,
    /// A deviation threshold was non-finite or negative.
    InvalidDeviationThreshold,
}

impl fmt::Display for PredictiveEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HistoryDepthTooSmall => write!(f, "history_depth must be at least 5"),
            Self::NumChannelsTooLarge => write!(
                f,
                "num_channels exceeds u16::MAX as usize + 1 (max addressable spike channels)"
            ),
            Self::InvalidDeviationThreshold => {
                write!(f, "deviation_threshold must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for PredictiveEncoderError {}

impl From<PredictiveEncoderError> for EncoderError {
    fn from(error: PredictiveEncoderError) -> Self {
        match error {
            PredictiveEncoderError::HistoryDepthTooSmall => {
                EncoderError::HistoryDepthTooSmall { minimum: 5 }
            }
            PredictiveEncoderError::NumChannelsTooLarge => EncoderError::NumChannelsTooLarge,
            PredictiveEncoderError::InvalidDeviationThreshold => EncoderError::NonNegativeFinite {
                parameter: "deviation_threshold",
            },
        }
    }
}

/// Encodes based on causal predictive error from expected values.
///
/// This encoder is best understood as an adaptive EWMA anomaly detector with
/// predictive-coding-style signed error spikes. It keeps per-channel history,
/// predicts the next sample from prior samples only, and fires a spike when the
/// signed prediction error is large enough. Positive errors emit
/// `polarity: true`; negative errors emit `polarity: false`.
///
/// # Mathematical Model
///
/// Tracks an exponentially weighted moving average of recent history means per
/// channel. The first five samples are a warm-up period: they update history and
/// initialize the prediction baseline but never emit spikes. After warm-up, each
/// input is evaluated against the predictor state formed before that input is
/// inserted, so the current observation cannot leak into its own prediction.
///
/// ```text
/// if history.len() < 5:
///     push value; initialize prediction when five samples are available; no spike
/// else:
///     prediction = threshold[i]
///     error = value - prediction
///     spike if |error| > threshold, with polarity = error >= 0
///     push value
///     threshold[i] = 0.9 * threshold[i] + 0.1 * mean(history[-5:])
/// ```
///
/// # When to Use
///
/// - Anomaly detection in sensor streams
/// - Learning patterns and detecting deviations
/// - Adaptive encoding that adjusts to baseline activity
///
/// # Parameters
///
/// - `history_depth`: Number of past values to track per channel
/// - `deviation_thresholds`: Vec of (threshold, spike_value) pairs
/// - `num_channels`: Number of input channels
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = PredictiveEncoder::try_new(8, vec![(0.5, 1)], 1)?;
/// // First five samples warm up without spikes.
/// for v in [1.0, 1.0, 1.0, 1.0, 1.0] {
///     assert!(enc.encode_step(&[v]).spikes.is_empty());
/// }
/// // A large jump after warm-up can emit a prediction-error spike.
/// let _ = enc.encode_step(&[3.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PredictiveEncoder {
    history: Vec<VecDeque<f32>>,
    thresholds: Vec<f32>,
    history_depth: usize,
    deviation_thresholds: Vec<(f32, u16)>,
}

impl PredictiveEncoder {
    /// Creates a new `PredictiveEncoder`.
    ///
    /// Retains the historical `PredictiveEncoderError` surface for source
    /// compatibility. Prefer [`try_new`](Self::try_new) for the unified
    /// [`EncoderError`] type used by other fallible constructors.
    ///
    /// # Errors
    ///
    /// - [`PredictiveEncoderError::HistoryDepthTooSmall`] if `history_depth < 5`
    /// - [`PredictiveEncoderError::NumChannelsTooLarge`] if `num_channels > u16::MAX as usize + 1`
    ///   (spike `channel` IDs are `u16`, so indices must stay in `0..=u16::MAX`)
    /// - [`PredictiveEncoderError::InvalidDeviationThreshold`] if any threshold is
    ///   non-finite or negative
    pub fn new(
        history_depth: usize,
        deviation_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Result<Self, PredictiveEncoderError> {
        Self::try_new(history_depth, deviation_thresholds, num_channels).map_err(
            |error| match error {
                EncoderError::HistoryDepthTooSmall { .. } => {
                    PredictiveEncoderError::HistoryDepthTooSmall
                }
                EncoderError::NumChannelsTooLarge => PredictiveEncoderError::NumChannelsTooLarge,
                EncoderError::NonNegativeFinite {
                    parameter: "deviation_threshold",
                } => PredictiveEncoderError::InvalidDeviationThreshold,
                other => panic!("unexpected EncoderError from PredictiveEncoder::try_new: {other}"),
            },
        )
    }

    /// Creates a new `PredictiveEncoder`, returning the unified [`EncoderError`].
    ///
    /// Prefer this over [`new`](Self::new) when propagating constructor failures
    /// alongside other encoders via `EncoderError`. Each `deviation_threshold`
    /// must be finite and non-negative (same rule as
    /// [`TemporalEncoder::try_new`](crate::encoders::TemporalEncoder::try_new)).
    pub fn try_new(
        history_depth: usize,
        deviation_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Result<Self, EncoderError> {
        if history_depth < 5 {
            return Err(EncoderError::HistoryDepthTooSmall { minimum: 5 });
        }
        for &(threshold, _) in &deviation_thresholds {
            crate::error::validate_non_negative_finite("deviation_threshold", threshold)?;
        }
        // encode_with_threshold_scale maps channel index → u16 via try_from.
        crate::error::validate_channel_count(num_channels)?;
        Ok(Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            thresholds: vec![0.0; num_channels],
            history_depth,
            deviation_thresholds,
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart.
    fn encode_with_threshold_scale_into(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
        sink: &mut dyn SpikeSink,
    ) {
        for (i, &value) in input.iter().enumerate() {
            if i >= self.history.len() {
                break;
            }
            let channel_history = &mut self.history[i];

            // Warm-up: history_depth is always >= 5, so no eviction can fire here.
            if channel_history.len() < 5 {
                channel_history.push_back(value);

                if channel_history.len() == 5 {
                    self.thresholds[i] = channel_history.iter().rev().take(5).sum::<f32>() / 5.0;
                }

                continue;
            }

            let prediction = self.thresholds[i];
            let error = value - prediction;
            let deviation = error.abs();

            for &(threshold, _spike_val) in self.deviation_thresholds.iter().rev() {
                if deviation > (threshold * threshold_scale).max(0.0) {
                    let Ok(channel) = u16::try_from(i) else {
                        break;
                    };
                    sink.push(SpikeEvent::at_step_start(channel, error >= 0.0));
                    break;
                }
            }

            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            let recent_avg = channel_history.iter().rev().take(5).sum::<f32>() / 5.0;
            self.thresholds[i] = 0.9 * self.thresholds[i] + 0.1 * recent_avg;
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

impl Encoder for PredictiveEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale(safe_input, 1.0)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_with_threshold_scale_into(input, 1.0, sink);
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale_into(safe_input, 1.0, sink);
    }

    /// One call is one tick; batch and streaming are identical here.
    ///
    /// A channel emits at most one spike per call, at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO), with polarity set by
    /// the sign of the prediction error. Ticks are dimensionless: one tick is
    /// one predicted sample.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
        for threshold in self.thresholds.iter_mut() {
            *threshold = 0.0;
        }
    }
}

impl ModulatedEncoder for PredictiveEncoder {
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
        self.encode_with_threshold_scale_into(safe_input, gains.sanitize().threshold_scale, sink);
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PredictiveEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::collections::VecDeque;

        #[derive(serde::Deserialize)]
        struct Helper {
            history: Vec<VecDeque<f32>>,
            thresholds: Vec<f32>,
            history_depth: usize,
            deviation_thresholds: Vec<(f32, u16)>,
        }

        let helper = Helper::deserialize(deserializer)?;

        if helper.history.len() != helper.thresholds.len() {
            return Err(serde::de::Error::custom(format!(
                "mismatched history length ({}) and thresholds length ({})",
                helper.history.len(),
                helper.thresholds.len()
            )));
        }

        // Mirror `new()`: spike channel IDs are u16 (indices 0..=u16::MAX).
        if helper.history.len() > u16::MAX as usize + 1 {
            return Err(serde::de::Error::custom(
                "num_channels exceeds u16::MAX as usize + 1 (max addressable spike channels)",
            ));
        }

        if helper.history_depth < 5 {
            return Err(serde::de::Error::custom("history_depth must be at least 5"));
        }

        // Match try_new: reject non-finite / negative deviation thresholds on load.
        for &(threshold, _) in &helper.deviation_thresholds {
            crate::error::validate_non_negative_finite("deviation_threshold", threshold)
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
            thresholds: helper.thresholds,
            history_depth: helper.history_depth,
            deviation_thresholds: helper.deviation_thresholds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_encoder_rejects_small_history_depth() {
        let err = PredictiveEncoder::new(4, vec![(2.0, 1)], 1).err();
        assert_eq!(err, Some(PredictiveEncoderError::HistoryDepthTooSmall));
        assert_eq!(
            PredictiveEncoderError::HistoryDepthTooSmall.to_string(),
            "history_depth must be at least 5"
        );
        assert!(PredictiveEncoder::new(5, vec![(2.0, 1)], 1).is_ok());
        assert!(PredictiveEncoder::new(0, vec![(2.0, 1)], 1).is_err());

        // try_new uses the unified EncoderError surface.
        assert_eq!(
            PredictiveEncoder::try_new(4, vec![(2.0, 1)], 1).err(),
            Some(EncoderError::HistoryDepthTooSmall { minimum: 5 })
        );
        assert_eq!(
            PredictiveEncoder::try_new(5, vec![(2.0, 1)], u16::MAX as usize + 2).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
        assert_eq!(
            PredictiveEncoder::try_new(5, vec![(f32::NAN, 1)], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "deviation_threshold"
            })
        );
        assert_eq!(
            PredictiveEncoder::try_new(5, vec![(-1.0, 1)], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "deviation_threshold"
            })
        );
        assert_eq!(
            PredictiveEncoder::new(5, vec![(-1.0, 1)], 1).err(),
            Some(PredictiveEncoderError::InvalidDeviationThreshold)
        );
        assert!(PredictiveEncoder::try_new(5, vec![(0.0, 1)], 1).is_ok());
    }

    #[test]
    fn test_predictive_encoder_num_channels_u16_range() {
        let max_ok = u16::MAX as usize + 1;
        let first_bad = max_ok + 1;

        // First rejected: fails before allocation / without panicking.
        assert_eq!(
            PredictiveEncoder::new(5, vec![(0.2, 1)], first_bad).err(),
            Some(PredictiveEncoderError::NumChannelsTooLarge)
        );
        assert!(
            PredictiveEncoderError::NumChannelsTooLarge
                .to_string()
                .contains("u16::MAX")
        );

        // Accepted maximum: every channel index is representable as u16.
        let encoder =
            PredictiveEncoder::new(5, vec![(0.2, 1)], max_ok).expect("max u16 channel count");
        assert_eq!(encoder.history.len(), max_ok);
        assert_eq!(encoder.thresholds.len(), max_ok);
    }

    #[test]
    fn test_predictive_encoder() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 1).expect("valid PredictiveEncoder");
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let output = encoder.encode(&[10.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_predictive_encoder_constant_signal_has_no_cold_start_burst() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(0.5, 1)], 1).expect("valid PredictiveEncoder");

        for _ in 0..16 {
            let output = encoder.encode(&[42.0]);
            assert!(output.spikes.is_empty());
        }
    }

    #[test]
    fn test_predictive_encoder_short_history_warms_up_without_spikes() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(0.5, 1)], 1).expect("valid PredictiveEncoder");

        for _ in 0..5 {
            let output = encoder.encode(&[10.0]);
            assert!(output.spikes.is_empty());
        }

        assert_eq!(encoder.history[0].len(), 5);
        assert_eq!(encoder.thresholds[0], 10.0);
    }

    #[test]
    fn test_predictive_encoder_positive_step_preserves_positive_polarity() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(1.0, 1)], 1).expect("valid PredictiveEncoder");
        for _ in 0..5 {
            assert!(encoder.encode(&[1.0]).spikes.is_empty());
        }

        let output = encoder.encode(&[4.0]);
        assert_eq!(output.spikes.len(), 1);
        assert!(output.spikes[0].polarity);
    }

    #[test]
    fn test_predictive_encoder_negative_step_preserves_negative_polarity() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(1.0, 1)], 1).expect("valid PredictiveEncoder");
        for _ in 0..5 {
            assert!(encoder.encode(&[4.0]).spikes.is_empty());
        }

        let output = encoder.encode(&[1.0]);
        assert_eq!(output.spikes.len(), 1);
        assert!(!output.spikes[0].polarity);
    }

    #[test]
    fn test_predictive_encoder_trend_uses_prior_prediction_before_update() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(0.75, 1)], 1).expect("valid PredictiveEncoder");
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            assert!(encoder.encode(&[value]).spikes.is_empty());
        }
        assert_eq!(encoder.thresholds[0], 3.0);

        let output = encoder.encode(&[6.0]);
        assert_eq!(output.spikes.len(), 1);
        assert!(output.spikes[0].polarity);
        assert_eq!(encoder.thresholds[0], 3.1);
    }

    #[test]
    fn test_predictive_encoder_reset() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode(&[1.0, 2.0]);
        }
        encoder.reset();
        assert!(encoder.history.iter().all(|h| h.is_empty()));
        assert!(encoder.thresholds.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn test_predictive_encoder_reset_restarts_warmup_without_spikes() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(0.5, 1)], 1).expect("valid PredictiveEncoder");
        for _ in 0..5 {
            assert!(encoder.encode(&[1.0]).spikes.is_empty());
        }
        assert_eq!(encoder.encode(&[10.0]).spikes.len(), 1);

        encoder.reset();

        for _ in 0..5 {
            assert!(encoder.encode(&[10.0]).spikes.is_empty());
        }
        assert_eq!(encoder.thresholds[0], 10.0);
    }

    #[test]
    fn test_predictive_encoder_multi_channel() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 3).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode(&[1.0, 2.0, 3.0]);
        }
        let output = encoder.encode(&[10.0, 20.0, 30.0]);
        // All channels should spike on large deviation
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_predictive_encoder_input_truncation() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            // 4 values but only 2 channels tracked — should truncate
            encoder.encode(&[1.0, 2.0, 3.0, 4.0]);
        }
        // Should not panic; only first 2 channels processed
        let output = encoder.encode(&[10.0, 20.0, 30.0, 40.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[test]
    fn test_predictive_encoder_step_input_truncation() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode_step(&[1.0, 2.0, 3.0]);
        }
        let output = encoder.encode_step(&[10.0, 20.0, 30.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[test]
    fn test_predictive_encoder_encode_with_modulators() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let mods = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        for _ in 0..5 {
            encoder.encode_with_modulators(&[1.0], &mods, &curves);
        }
        let output = encoder.encode_with_modulators(&[5.0], &mods, &curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_modulators_reduce_threshold() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let modulators = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..5 {
            encoder.encode(&[1.0]);
        }
        assert!(encoder.encode(&[5.0]).spikes.is_empty());

        encoder.reset();

        for _ in 0..5 {
            encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_encode_with_modulators_truncate() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let mods = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        for _ in 0..5 {
            encoder.encode_with_modulators(&[1.0, 2.0], &mods, &curves);
        }
        let output = encoder.encode_with_modulators(&[5.0, 6.0], &mods, &curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_step_shorter_input() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode_step(&[1.0]);
        }
        let output = encoder.encode_step(&[10.0]);
        assert!(!output.spikes.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_predictive_serde_history_channel_too_long() {
        let json = r#"{
            "history": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "thresholds": [0.0],
            "history_depth": 5,
            "deviation_thresholds": []
        }"#;
        let res: Result<PredictiveEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_predictive_serde_rejects_too_many_channels() {
        let max_ok = u16::MAX as usize + 1;
        let first_bad = max_ok + 1;

        // First rejected: same ceiling as `new()` (untrusted saved state cannot bypass).
        let history: Vec<Vec<f32>> = vec![vec![]; first_bad];
        let thresholds = vec![0.0f32; first_bad];
        let value = serde_json::json!({
            "history": history,
            "thresholds": thresholds,
            "history_depth": 5,
            "deviation_thresholds": [[0.2, 1]],
        });
        let res: Result<PredictiveEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
        let err = res.err().unwrap().to_string();
        assert!(
            err.contains("u16::MAX") || err.contains("num_channels"),
            "unexpected error: {err}"
        );

        // Accepted maximum boundary still deserializes.
        let history_ok: Vec<Vec<f32>> = vec![vec![]; max_ok];
        let thresholds_ok = vec![0.0f32; max_ok];
        let value_ok = serde_json::json!({
            "history": history_ok,
            "thresholds": thresholds_ok,
            "history_depth": 5,
            "deviation_thresholds": [[0.2, 1]],
        });
        let enc: PredictiveEncoder =
            serde_json::from_value(value_ok).expect("max channel count deserializes");
        assert_eq!(enc.history.len(), max_ok);
        assert_eq!(enc.thresholds.len(), max_ok);
    }
}
