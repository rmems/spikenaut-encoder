use crate::prelude::*;

/// Encodes analog values as phase-locked spikes within a repeating oscillation cycle
///
/// Each input channel produces at most one positive spike per call, placed within
/// the ongoing cycle according to the normalized input value. Higher values map to
/// later phase bins, so ordering is stable within a call.
///
/// # Time semantics
///
/// The background oscillation advances **one tick per call**, while a single call
/// can place a spike anywhere in the `cycle_steps`-tick cycle — this is the one
/// encoder here whose windows overlap
/// ([`TimeModel::is_overlapping`](crate::time::TimeModel::is_overlapping) is
/// `true`). Like every encoder in this crate, the emitted `timestamp` is the
/// **call-relative** phase offset in `0..cycle_steps`; absolute phase time is
/// `cursor.absolute(offset)`, and the position within the cycle is
/// `cursor.absolute(offset) % cycle_steps`.
///
/// [`current_phase`](Self::current_phase) exposes the encoder's own cycle
/// counter for callers that track the oscillation directly rather than through a
/// [`TimeCursor`](crate::time::TimeCursor).
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = PhaseEncoder::try_new(16, (0.0, 1.0))?;
/// let mut cursor = TimeCursor::new(enc.time_model());
///
/// let out = enc.encode(&[0.0, 1.0]);
/// assert_eq!(out.spikes.len(), 2);
/// assert_eq!(out.spikes[0].timestamp, 0); // low value → early in the cycle
/// assert_eq!(out.spikes[1].timestamp, 15); // high value → late in the cycle
///
/// cursor.advance(); // one tick of background oscillation per call
/// let next = enc.encode(&[0.0]);
/// assert_eq!(next.spikes[0].timestamp, 0); // still cycle-relative
/// assert_eq!(cursor.absolute(next.spikes[0].timestamp), 1); // absolute phase time
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PhaseEncoder {
    cycle_steps: u64,
    range: (f32, f32),
    current_phase: u64,
}

/// Validates `cycle_steps` and `range`, returning an error message if invalid
///
/// Shared by both `PhaseEncoder::new` (which panics on failure) and the
/// `Deserialize` impl (which surfaces the message as a deserialization error)
fn validate_params(cycle_steps: u64, range: (f32, f32)) -> Result<(), EncoderError> {
    if cycle_steps == 0 {
        return Err(EncoderError::WindowMustBePositive {
            parameter: "cycle_steps",
        });
    }
    crate::error::validate_range("range", range)
}

impl PhaseEncoder {
    /// Creates a new `PhaseEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if `cycle_steps == 0` or if range bounds are non-finite or `range.0 >= range.1`.
    pub fn new(cycle_steps: u64, range: (f32, f32)) -> Self {
        Self::try_new(cycle_steps, range).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a new `PhaseEncoder`, returning an [`EncoderError`] for invalid configuration.
    pub fn try_new(cycle_steps: u64, range: (f32, f32)) -> Result<Self, EncoderError> {
        validate_params(cycle_steps, range)?;
        Ok(Self {
            cycle_steps,
            range,
            current_phase: 0,
        })
    }

    fn normalize(&self, value: f32) -> f64 {
        // Use f64 to prevent overflow for valid f32 ranges (e.g., f32::MIN..f32::MAX).
        let clamped = value.clamp(self.range.0, self.range.1) as f64;
        let lo = self.range.0 as f64;
        let hi = self.range.1 as f64;
        (clamped - lo) / (hi - lo)
    }

    fn phase_offset(&self, normalized: f64) -> u64 {
        ((normalized * self.cycle_steps as f64).floor() as u64).min(self.cycle_steps - 1)
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Emits the current cycle only — advancing the background phase stays with
    /// the caller, exactly as in the returning paths. Every public encoding
    /// path on this encoder routes through here, so the returning and
    /// sink-based APIs cannot drift apart.
    fn encode_current_cycle_into<S: SpikeSink + ?Sized>(&self, input: &[f32], sink: &mut S) {
        for (channel, &value) in input.iter().enumerate() {
            // Non-finite inputs are invalid readings — skip rather than emit a
            // misleading phase-0 spike (NaN as u64 saturates to 0).
            if !value.is_finite() {
                continue;
            }

            let Ok(channel_u16) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };

            // Call-relative phase offset: absolute phase time is the caller's
            // cursor origin plus this offset (see the crate time contract).
            let phase_offset = self.phase_offset(self.normalize(value));
            sink.push(SpikeEvent::new(channel_u16, phase_offset, true));
        }
    }

    fn encode_current_cycle(&self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_current_cycle_into(input, &mut output.spikes);
        output
    }

    fn advance_phase(&mut self) {
        self.current_phase = self.current_phase.saturating_add(1);
    }

    /// Absolute tick of the background oscillation, advanced once per call.
    ///
    /// Equivalent to the origin of a [`TimeCursor`](crate::time::TimeCursor)
    /// driven by this encoder's [`time_model`](Encoder::time_model), and useful
    /// when a caller wants the cycle position without keeping its own cursor:
    /// `(encoder.current_phase() + spike.timestamp.ticks()) % cycle_steps`.
    ///
    /// Read it **before** the emitting call. Every encode call advances the
    /// counter after producing its output, so a value read afterward belongs to
    /// the *next* call and would place the spikes one tick late.
    #[inline]
    pub const fn current_phase(&self) -> u64 {
        self.current_phase
    }

    /// Ticks in one full oscillation cycle.
    #[inline]
    pub const fn cycle_steps(&self) -> u64 {
        self.cycle_steps
    }

    /// Gain-scaled counterpart of
    /// [`encode_current_cycle_into`](Self::encode_current_cycle_into).
    fn encode_current_cycle_with_sensitivity_scale_into<S: SpikeSink + ?Sized>(
        &self,
        input: &[f32],
        sensitivity_scale: f32,
        sink: &mut S,
    ) {
        // Guard: zero or non-finite sensitivity collapses the range, suppressing all output.
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return;
        }

        // Use f64 to prevent overflow for valid f32 ranges and scales.
        let lo = self.range.0 as f64;
        let hi = lo + (self.range.1 as f64 - lo) * (sensitivity_scale as f64);

        for (channel, &value) in input.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }

            let Ok(channel_u16) = u16::try_from(channel) else {
                break;
            };

            let normalized = ((value as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let phase_offset = self.phase_offset(normalized);
            sink.push(SpikeEvent::new(channel_u16, phase_offset, true));
        }
    }

    fn encode_current_cycle_with_sensitivity_scale(
        &self,
        input: &[f32],
        sensitivity_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_current_cycle_with_sensitivity_scale_into(
            input,
            sensitivity_scale,
            &mut output.spikes,
        );
        output
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

impl Encoder for PhaseEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        // Streaming and batch modes share the same phase-step semantics for
        // this encoder: each call advances the background oscillation by one.
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.encode_current_cycle_into(input, sink));
        self.advance_phase();
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_into(input, sink);
    }

    /// An overlapping model: one tick of oscillation per call, `cycle_steps` of
    /// reach.
    ///
    /// Batch and streaming behave identically — both advance the background
    /// phase by exactly one tick — so a caller advances its cursor by 1 per call
    /// while a single call can place spikes up to `cycle_steps - 1` ticks ahead.
    /// Ticks are dimensionless: `cycle_steps` divides a cycle, and the caller
    /// chooses what a cycle lasts.
    fn time_model(&self) -> TimeModel {
        TimeModel::overlapping(1, self.cycle_steps)
    }

    fn reset(&mut self) {
        self.current_phase = 0;
    }
}

impl ModulatedEncoder for PhaseEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let output = self
            .encode_current_cycle_with_sensitivity_scale(input, gains.sanitize().sensitivity_scale);
        self.advance_phase();
        output
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let sensitivity_scale = gains.sanitize().sensitivity_scale;
        crate::sink::through_chunks(sink, |sink| {
            self.encode_current_cycle_with_sensitivity_scale_into(input, sensitivity_scale, sink)
        });
        self.advance_phase();
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PhaseEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            cycle_steps: u64,
            range: (f32, f32),
            #[serde(default)]
            current_phase: u64,
        }

        let helper = Helper::deserialize(deserializer)?;

        validate_params(helper.cycle_steps, helper.range).map_err(serde::de::Error::custom)?;

        Ok(Self {
            cycle_steps: helper.cycle_steps,
            range: helper.range,
            current_phase: helper.current_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wide_range_normalizes_without_nan() {
        let mut encoder = PhaseEncoder::new(8, (f32::MIN, f32::MAX));
        let output = encoder.encode(&[f32::MAX]);
        assert_eq!(output.spikes.len(), 1);
        // f32::MAX maps to the last phase bin, not NaN → phase 0.
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_phase_mapping_clamps_and_quantizes() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 10.0));

        let output = encoder.encode(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let timestamps: Vec<u64> = output
            .spikes
            .iter()
            .map(|spike| spike.timestamp.ticks())
            .collect();
        let polarities: Vec<bool> = output.spikes.iter().map(|spike| spike.polarity).collect();

        assert_eq!(timestamps, vec![0, 0, 4, 7, 7]);
        assert_eq!(polarities, vec![true; 5]);
    }

    #[test]
    fn test_phase_offsets_are_call_relative() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        // The emitted offset is the position within the cycle, so an unchanged
        // input yields an unchanged offset no matter how far time has advanced.
        for _ in 0..6 {
            assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 0);
        }
        assert_eq!(encoder.encode_step(&[1.0]).spikes[0].timestamp, 3);
    }

    #[test]
    fn test_phase_advances_one_tick_per_call() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));
        let mut cursor = TimeCursor::new(encoder.time_model());

        // Both modes advance the background oscillation by exactly one tick.
        for expected_phase in 0..4 {
            assert_eq!(encoder.current_phase(), expected_phase);
            assert_eq!(cursor.origin(), expected_phase);
            let output = if expected_phase % 2 == 0 {
                encoder.encode(&[0.0])
            } else {
                encoder.encode_step(&[0.0])
            };
            // Absolute phase time is the cursor origin plus the emitted offset.
            assert_eq!(cursor.absolute(output.spikes[0].timestamp), expected_phase);
            cursor.advance();
        }

        // Cycle position stays recoverable from absolute time.
        let output = encoder.encode(&[0.5]);
        let absolute = cursor.absolute(output.spikes[0].timestamp);
        assert_eq!(absolute, 6);
        assert_eq!(absolute % encoder.cycle_steps(), 2);
    }

    #[test]
    fn test_within_call_ordering_preserved_after_phase_advance() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let mut cursor = TimeCursor::new(encoder.time_model());
        // Advance near the end of a modular cycle so a wrap would reorder.
        for _ in 0..6 {
            encoder.encode(&[0.0]);
            cursor.advance();
        }
        let output = encoder.encode(&[0.125, 0.375]); // offsets 1 and 3
        let offsets: Vec<u64> = output
            .spikes
            .iter()
            .map(|spike| spike.timestamp.ticks())
            .collect();
        assert_eq!(offsets, vec![1, 3]);

        // 6+1=7, 6+3=9 on the caller's timeline — strictly ordered, no modular
        // wrap inversion.
        let absolute: Vec<u64> = cursor.absolute_times(&output.spikes).collect();
        assert_eq!(absolute, vec![7, 9]);
        assert!(absolute[0] < absolute[1]);
    }

    #[test]
    fn test_reset_restores_initial_phase() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));

        encoder.encode(&[0.0]);
        encoder.encode(&[0.0]);
        assert_eq!(encoder.current_phase(), 2);
        encoder.reset();
        assert_eq!(encoder.current_phase(), 0);

        let output = encoder.encode(&[1.0]);
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_empty_input_returns_no_spikes() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());
        // An empty call still advances the background oscillation.
        assert_eq!(encoder.current_phase(), 1);

        let next_output = encoder.encode(&[0.0]);
        assert_eq!(next_output.spikes[0].timestamp, 0);
        assert_eq!(encoder.current_phase(), 2);
    }

    #[test]
    fn test_phase_time_model_is_overlapping() {
        let encoder = PhaseEncoder::new(16, (0.0, 1.0));
        let model = encoder.time_model();

        assert_eq!(model.step_ticks(), 1);
        assert_eq!(model.span_ticks(), 16);
        assert!(model.is_overlapping());
        assert!(model.timebase().is_none());
    }

    #[test]
    fn test_nan_input_skips_channel() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let output = encoder.encode(&[0.0, f32::NAN, 1.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(output.spikes[1].channel, 2);
    }

    #[test]
    #[should_panic(expected = "cycle_steps must be greater than 0")]
    fn test_zero_cycle_steps_rejected() {
        let _ = PhaseEncoder::new(0, (0.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "range must be finite and min must be less than max")]
    fn test_invalid_range_rejected() {
        let _ = PhaseEncoder::new(8, (1.0, 1.0));
    }

    #[test]
    fn test_encode_step_matches_encode() {
        let input = [2.5, 7.5];
        let mut encode_encoder = PhaseEncoder::new(8, (0.0, 10.0));
        let mut step_encoder = PhaseEncoder::new(8, (0.0, 10.0));

        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_deserialize_rejects_zero_cycle_steps() {
        let json = r#"{"cycle_steps":0,"range":[0.0,1.0],"current_phase":0}"#;
        let err = serde_json::from_str::<PhaseEncoder>(json).unwrap_err();
        assert!(err.to_string().contains("cycle_steps"));
    }

    #[test]
    fn test_encode_with_modulators_identity() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let plain = encoder.encode(&[0.5]);
        let mut encoder2 = PhaseEncoder::new(8, (0.0, 1.0));
        let modulated = encoder2.encode_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(plain.spikes[0].timestamp, modulated.spikes[0].timestamp);
    }

    #[test]
    fn test_encode_with_modulators_sensitivity_scale() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (0.5, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        // sensitivity_scale = 0.5, range = (0.0, 0.5)
        // value 0.5 maps to normalized 1.0, phase_offset = 7
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_encode_step_with_modulators_matches_encode() {
        let input = [0.5];
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let mut encoder1 = PhaseEncoder::new(8, (0.0, 1.0));
        let mut encoder2 = PhaseEncoder::new(8, (0.0, 1.0));

        let batch = encoder1.encode_with_modulators(&input, &mods, &curves);
        let step = encoder2.encode_step_with_modulators(&input, &mods, &curves);

        assert_eq!(batch, step);
    }

    #[test]
    fn test_encode_with_modulators_zero_sensitivity_suppresses() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (0.0, 0.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_encode_with_modulators_nan_input_skips() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (1.0, 1.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.0, f32::NAN, 1.0], &mods, &curves);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(output.spikes[1].channel, 2);
    }
    #[test]
    fn test_phase_encoder_try_new_validation() {
        assert_eq!(
            PhaseEncoder::try_new(0, (0.0, 1.0)).err(),
            Some(EncoderError::WindowMustBePositive {
                parameter: "cycle_steps"
            })
        );
        assert_eq!(
            PhaseEncoder::try_new(1, (1.0, 1.0)).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
    }
}
