use crate::prelude::*;

/// Encodes analog values as spike rates based on input intensity.
///
/// Each input channel is mapped to a firing rate between `base_rate` and `max_rate`.
/// In batch mode (`encode`), each call generates independent probabilistic spikes.
/// In streaming mode (`encode_step`), accumulates expected spikes and fires deterministically
/// when the accumulated value exceeds a threshold per channel.
///
/// # Mathematical Model
///
/// For batch encoding:
/// ```text
/// rate_hz = base_rate + normalized * (max_rate - base_rate)
/// probability = 1 - exp(-rate_hz * dt_seconds)
/// spike if random() < probability
/// ```
///
/// For streaming (`encode_step`):
/// ```text
/// rate_hz = base_rate + normalized[i] * (max_rate - base_rate)
/// accumulator[i] += rate_hz * dt_seconds
/// spike if accumulator[i] >= 1.0 (then accumulator -= 1.0)
/// ```
///
/// # When to Use
///
/// - Converting continuous sensor values to spike rates
/// - Poisson-like spike generation with controllable average rates
/// - Real-time encoding where spike timing follows input intensity
///
/// # Parameters
///
/// - `base_rate`: Minimum firing rate in hertz (Hz) when input is at range minimum
/// - `max_rate`: Maximum firing rate in hertz (Hz) when input is at range maximum
/// - `range`: Tuple of (min, max) input values
/// - `dt_seconds`: Duration, in seconds, represented by each encode step
///
/// # Migration
///
/// [`RateEncoder::new`] keeps the previous constructor shape and uses
/// `dt_seconds = 0.1`, which preserves the old deterministic `/ 10.0`
/// increment for unit rates. Prefer [`RateEncoder::try_new`] for new code that
/// wants explicit time-step configuration and validation.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010)?;
/// let out = enc.encode(&[0.0, 0.5, 1.0]);
/// // Stochastic batch mode: at most one spike per channel.
/// assert!(out.spikes.len() <= 3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
    dt_seconds: f32,
    /// Fractional phase per channel, kept in `[0, 1)`.
    ///
    /// Serialized as `accumulators` for backward compatibility with earlier
    /// checkpoints that stored a single combined float per channel.
    #[cfg_attr(feature = "serde", serde(rename = "accumulators"))]
    phases: Vec<f64>,
    /// Exact whole-spike backlog per channel (drainable past f64's `2^53` cliff).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pending_spikes: Vec<u64>,
}

impl RateEncoder {
    /// Compatibility time step used by [`RateEncoder::new`].
    ///
    /// A 100 ms step makes `rate_hz * dt_seconds` equal to the previous
    /// deterministic `/ 10.0` increment for the same rate value.
    pub const DEFAULT_DT_SECONDS: f32 = 0.1;

    /// Creates a rate encoder with the compatibility `dt_seconds = 0.1`.
    ///
    /// Prefer [`RateEncoder::try_new`] when selecting an explicit sampling
    /// interval for new code.
    ///
    /// # Panics
    ///
    /// Panics if rates or range are invalid (`dt_seconds` is always the valid
    /// default `0.1`, so callers cannot panic via the time step).
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self::try_new(base_rate, max_rate, range, Self::DEFAULT_DT_SECONDS)
            .expect("invalid RateEncoder configuration")
    }

    /// Creates a rate encoder with an explicit time step in seconds.
    ///
    /// Rates must be finite and non-negative with `base_rate <= max_rate`.
    /// `dt_seconds` must be finite and strictly positive. Range must be a
    /// non-degenerate finite f32 span (bounds finite, ordered, and
    /// `max - min` finite in f32).
    pub fn try_new(
        base_rate: f32,
        max_rate: f32,
        range: (f32, f32),
        dt_seconds: f32,
    ) -> Result<Self, EncoderError> {
        crate::error::validate_non_negative_finite("base_rate", base_rate)?;
        crate::error::validate_non_negative_finite("max_rate", max_rate)?;
        if base_rate > max_rate {
            return Err(EncoderError::RateOrder);
        }
        crate::error::validate_range_f32_span("range", range)?;
        Self::validate_dt_seconds(dt_seconds)?;
        Ok(Self {
            base_rate,
            max_rate,
            range,
            dt_seconds,
            phases: Vec::new(),
            pending_spikes: Vec::new(),
        })
    }

    /// Returns the configured time step in seconds.
    pub fn dt_seconds(&self) -> f32 {
        self.dt_seconds
    }

    pub fn default_dt_seconds() -> f32 {
        Self::DEFAULT_DT_SECONDS
    }

    fn validate_dt_seconds(dt_seconds: f32) -> Result<(), EncoderError> {
        if dt_seconds.is_finite() && dt_seconds > 0.0 {
            Ok(())
        } else {
            Err(EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds",
            })
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    /// Effective firing rate in Hz after range mapping and gain scale.
    ///
    /// Always returns a finite non-negative `f32`: non-finite inputs silence,
    /// and positive overflow saturates to `f32::MAX` (so batch encoding does not
    /// treat strong modulated rates as silent via non-finite rejection).
    fn effective_rate_hz(&self, value: f32, rate_scale: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        let normalized = f64::from(self.normalize(value));
        let base = f64::from(self.base_rate)
            + normalized * (f64::from(self.max_rate) - f64::from(self.base_rate));
        let rate = base * f64::from(rate_scale);
        // NaN comparisons are false → 0.0; +∞ > 0 → MAX; finite values clamp.
        if !rate.is_finite() {
            return if rate > 0.0 { f32::MAX } else { 0.0 };
        }
        rate.clamp(0.0, f64::from(f32::MAX)) as f32
    }

    fn ensure_accumulators(&mut self, num_channels: usize) {
        if self.phases.len() < num_channels {
            self.phases.resize(num_channels, 0.0);
            self.pending_spikes.resize(num_channels, 0);
        }
    }

    /// Split a finite non-negative value into whole spikes + fractional phase.
    fn split_whole_and_frac(value: f64) -> (u64, f64) {
        debug_assert!(value.is_finite() && value >= 0.0);
        if value < 1.0 {
            return (0, value);
        }
        if value >= u64::MAX as f64 {
            return (u64::MAX, 0.0);
        }
        let whole = value.trunc() as u64;
        let frac = (value - whole as f64).clamp(0.0, 1.0 - f64::EPSILON);
        (whole, frac)
    }

    /// Apply a finite non-negative expected-spike increment to channel state.
    fn apply_streaming_increment(&mut self, channel_idx: usize, increment: f64) {
        if increment <= 0.0 {
            return;
        }
        // Cap so phase + increment stays finite and within the u64 backlog range.
        let sum = (self.phases[channel_idx] + increment).min(u64::MAX as f64);
        let (whole, frac) = Self::split_whole_and_frac(sum);
        self.pending_spikes[channel_idx] = self.pending_spikes[channel_idx].saturating_add(whole);
        self.phases[channel_idx] = frac;
    }

    /// Batch spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public batch encoding path routes through here, so the returning
    /// and sink-based APIs cannot drift apart.
    fn encode_with_rate_scale_into(
        &mut self,
        input: &[f32],
        rate_scale: f32,
        sink: &mut dyn SpikeSink,
    ) {
        if input.is_empty() {
            return;
        }
        // Match PopulationEncoder: non-finite or non-positive scales fully silence.
        // Avoids NaN probabilities that would silently never spike.
        if !rate_scale.is_finite() || rate_scale <= 0.0 {
            return;
        }

        let mut rng = rand::rng();
        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let rate = self.effective_rate_hz(value, rate_scale);
            let probability = crate::poisson::probability_from_rate_hz(rate, self.dt_seconds);

            if crate::rng::gen_unit_f32_with_rng(&mut rng) < probability {
                sink.push(SpikeEvent::at_step_start(channel, true));
            }
        }
    }

    fn encode_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_with_rate_scale_into(input, rate_scale, &mut output.spikes);
        output
    }

    /// Cap on spikes emitted per channel per streaming step.
    ///
    /// Bounds allocation when `rate_hz * dt_seconds` is huge. Remaining whole
    /// spikes stay in the exact `u64` pending queue and drain on later
    /// `encode_step` calls (no permanent loss of expected spike count, and no
    /// stall past f64's `2^53` integer cliff where `acc -= 1.0` would no-op).
    /// Non-finite increments are skipped so the emission loop always terminates.
    const MAX_SPIKES_PER_CHANNEL_PER_STEP: usize = 1024;

    /// Expected spikes for one streaming step (`rate_hz * dt_seconds` in f64).
    ///
    /// Always finite and non-negative; saturates at `u64::MAX`. Using f64 avoids
    /// silent drops when the f32 product would overflow to `+inf`.
    fn streaming_increment(&self, value: f32, rate_scale: f32) -> f64 {
        let rate_hz = self.effective_rate_hz(value, rate_scale);
        if rate_hz <= 0.0 {
            return 0.0;
        }
        // rate_hz and dt_seconds are finite → product is finite in f64.
        (f64::from(rate_hz) * f64::from(self.dt_seconds)).clamp(0.0, u64::MAX as f64)
    }

    fn emit_capped_channel_spikes(
        &mut self,
        channel: u16,
        channel_idx: usize,
        sink: &mut dyn SpikeSink,
    ) {
        let pending = self.pending_spikes[channel_idx];
        if pending == 0 {
            return;
        }
        let emit = pending.min(Self::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64) as usize;
        if emit > 1 {
            // Only worth a hint for a drained backlog; the common single-spike
            // case skips the virtual call entirely.
            sink.reserve(emit);
        }
        // Coincident repeats: contiguous, same offset, mutually unordered — the
        // run length is this channel's spike count for the step.
        for _ in 0..emit {
            sink.push(SpikeEvent::at_step_start(channel, true));
        }
        self.pending_spikes[channel_idx] = pending - emit as u64;
        // Any remaining whole spikes stay queued for subsequent steps.
    }

    fn rate_scale_is_active(rate_scale: f32) -> bool {
        rate_scale.is_finite() && rate_scale > 0.0
    }

    /// Streaming spike-emitting core: writes straight into `sink`.
    ///
    /// Allocation-free once the per-channel accumulators are sized, which
    /// happens on the first call for a given channel count.
    fn encode_step_with_rate_scale_into(
        &mut self,
        input: &[f32],
        rate_scale: f32,
        sink: &mut dyn SpikeSink,
    ) {
        if input.is_empty() {
            return;
        }

        self.ensure_accumulators(input.len());
        let active = Self::rate_scale_is_active(rate_scale);

        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                break;
            };
            if !active {
                // Documented silence (`firing_rate_scale = 0`): no spikes and
                // drop backlog so a later non-zero gain cannot burst old debt.
                self.pending_spikes[i] = 0;
                self.phases[i] = 0.0;
                continue;
            }
            let increment = self.streaming_increment(value, rate_scale);
            self.apply_streaming_increment(i, increment);
            self.emit_capped_channel_spikes(channel, i, sink);
        }
    }

    fn encode_step_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_step_with_rate_scale_into(input, rate_scale, &mut output.spikes);
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
    ///
    /// Uses the internal accumulator-based rate scale path for streaming.
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
impl<'de> serde::Deserialize<'de> for RateEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            base_rate: f32,
            max_rate: f32,
            range: (f32, f32),
            #[serde(default = "RateEncoder::default_dt_seconds")]
            dt_seconds: f32,
            /// Legacy combined float (phase + whole spikes) and/or fractional phase.
            #[serde(default)]
            accumulators: Vec<f64>,
            /// Exact whole-spike backlog (new format). Folded with any whole part
            /// still present in `accumulators` for forward/backward compatibility.
            #[serde(default)]
            pending_spikes: Vec<u64>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let mut encoder = Self::try_new(
            helper.base_rate,
            helper.max_rate,
            helper.range,
            helper.dt_seconds,
        )
        .map_err(serde::de::Error::custom)?;
        // Allow values >= 1.0 in `accumulators` (legacy combined representation).
        // Reject only non-finite or negative state.
        if helper
            .accumulators
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(serde::de::Error::custom(
                "accumulators must be finite and non-negative",
            ));
        }
        let n = helper.accumulators.len().max(helper.pending_spikes.len());
        encoder.phases = vec![0.0; n];
        encoder.pending_spikes = vec![0; n];
        for i in 0..n {
            let combined = helper.accumulators.get(i).copied().unwrap_or(0.0);
            let (whole_from_acc, phase) = Self::split_whole_and_frac(combined);
            let pending = helper.pending_spikes.get(i).copied().unwrap_or(0);
            encoder.phases[i] = phase;
            encoder.pending_spikes[i] = pending.saturating_add(whole_from_acc);
        }
        Ok(encoder)
    }
}

impl Encoder for RateEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_rate_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step_with_rate_scale(input, 1.0)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_with_rate_scale_into(input, 1.0, sink);
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_step_with_rate_scale_into(input, 1.0, sink);
    }

    /// One call is one tick, sized by `dt_seconds`.
    ///
    /// Every spike lands at [`TickOffset::ZERO`](crate::time::TickOffset::ZERO)
    /// in both modes — a Poisson draw says *how many* spikes fall in the step,
    /// not where within it. Streaming can therefore emit several coincident
    /// spikes for one channel; per the crate time contract they are contiguous
    /// and mutually unordered, so the run length is a spike count.
    ///
    /// Unlike the other step-wise encoders this one is calibrated in physical
    /// time, so it reports a [`Timebase`](crate::time::Timebase) of
    /// `dt_seconds`. The timebase is omitted when `dt_seconds` has no whole-
    /// nanosecond representation — it rounds below one nanosecond, or exceeds
    /// the `u64` nanosecond range — since `try_new` accepts any finite positive
    /// interval. Ticks are still well defined in that case; only their physical
    /// duration is unavailable.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
            .with_timebase_opt(Timebase::try_from_seconds(f64::from(self.dt_seconds)).ok())
    }

    fn reset(&mut self) {
        self.phases.fill(0.0);
        self.pending_spikes.fill(0);
    }
}

impl ModulatedEncoder for RateEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_rate_scale(input, gains.sanitize().firing_rate_scale)
    }

    fn encode_step_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_step_with_rate_scale(input, gains.sanitize().firing_rate_scale)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_rate_scale_into(input, gains.sanitize().firing_rate_scale, sink);
    }

    fn encode_step_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_step_with_rate_scale_into(input, gains.sanitize().firing_rate_scale, sink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_encoder_basic() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [0.0, 50.0, 100.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
    }

    #[test]
    fn test_rate_encoder_encode_step() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        // max_rate 10.0 -> increment = (0.0 + 1.0 * 10.0) / 10.0 = 1.0
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(output.spikes.len(), 1);

        let output2 = encoder.encode_step(&[0.5]);
        // 0.5 * 10.0 / 10.0 = 0.5 increment
        assert_eq!(output2.spikes.len(), 0);
        let output3 = encoder.encode_step(&[0.5]);
        // another 0.5 -> 1.0 -> spike
        assert_eq!(output3.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_empty_input() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input: [f32; 0] = [];
        let output = encoder.encode(&input);
        assert_eq!(output.spikes.len(), 0);
        let output_step = encoder.encode_step(&input);
        assert_eq!(output_step.spikes.len(), 0);
    }

    #[test]
    fn test_rate_encoder_single_channel() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 1);
    }

    #[test]
    fn test_rate_encoder_below_min() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [-50.0, -100.0, -1.0];
        let output = encoder.encode(&input);
        assert!(
            output.spikes.is_empty(),
            "Below-min inputs should produce no spikes"
        );
    }

    #[test]
    fn test_rate_encoder_above_max() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [150.0, 200.0, 101.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
        for spike in &output.spikes {
            assert!(u32::from(spike.channel) < 3);
        }
    }

    #[test]
    fn test_rate_encoder_reset_does_not_panic() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5; 10];
        encoder.encode(&input);
        encoder.reset();
        encoder.encode(&input);
    }

    #[test]
    fn test_rate_encoder_never_panics() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let inputs: [&[f32]; 4] = [&[], &[0.0], &[50.0, 100.0], &[f32::MIN, f32::MAX]];
        for input in inputs {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.encode(input)));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_rate_encoder_modulated_step_scales_firing_rate() {
        let mut encoder = RateEncoder::new(0.0, 5.0, (0.0, 1.0));
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };

        let baseline = encoder.encode_step(&[1.0]);
        assert!(baseline.spikes.is_empty());

        encoder.reset();

        let boosted = encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        assert_eq!(boosted.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_encode_with_modulators() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };

        // Use the deterministic streaming path: dopamine doubles the 10 Hz max
        // rate to 20 Hz, so at dt=0.1s the accumulator advances by 2.0 and emits
        // two spikes. Batch `encode_with_modulators` is stochastic (p ≈ 0.865)
        // and flaky under CI, so it is not used here.
        let boosted = encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        assert_eq!(boosted.spikes.len(), 2);
        assert!(boosted.spikes.iter().all(|s| s.channel == 0));

        // Baseline (identity gains) advances by 1.0 and emits a single spike.
        let mut baseline = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let identity = baseline.encode_step_with_modulators(
            &[1.0],
            &NeuroModulators::default(),
            &NeuromodulatorGainCurves::default(),
        );
        assert_eq!(identity.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_step_shorter_input() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        // Grow accumulators to two channels, then step with a shorter slice so only
        // channel 0 is updated; channel 1 state is left untouched.
        let _ = encoder.encode_step(&[0.0, 0.0]);
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(output.spikes.len(), 1);
        // Channel 1 still at zero accumulation: another zero-only step on both
        // channels must not invent a ch1 spike.
        let quiet = encoder.encode_step(&[0.0, 0.0]);
        assert!(quiet.spikes.is_empty());
    }

    #[test]
    fn test_rate_encoder_zero_rate_scale_never_accumulates() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        for _ in 0..10_000 {
            let output = encoder.encode_step_with_rate_scale(&[1.0], 0.0);
            assert!(
                output.spikes.is_empty(),
                "zero firing-rate scale must fully silence streaming output"
            );
        }
    }

    #[test]
    fn test_rate_encoder_non_finite_rate_scale_silences() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            let batch = encoder.encode_with_rate_scale(&[1.0], scale);
            assert!(
                batch.spikes.is_empty(),
                "non-finite/negative rate_scale ({scale}) must silence batch encode"
            );
            let step = encoder.encode_step_with_rate_scale(&[1.0], scale);
            assert!(
                step.spikes.is_empty(),
                "non-finite/negative rate_scale ({scale}) must silence streaming encode"
            );
        }
        // Accumulators must not be poisoned: a normal step after NaN still works.
        encoder.reset();
        let recovered = encoder.encode_step_with_rate_scale(&[1.0], 1.0);
        assert_eq!(recovered.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_try_new_validation() {
        let dt = RateEncoder::DEFAULT_DT_SECONDS;
        assert_eq!(
            RateEncoder::try_new(f32::NAN, 1.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "base_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, f32::INFINITY, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "max_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(-5.0, 10.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "base_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(2.0, 1.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::RateOrder)
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 1.0, (1.0, 1.0), dt).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 1.0, (f32::MIN, f32::MAX), dt).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.0).err(),
            Some(EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds"
            })
        );
    }

    #[test]
    fn test_rate_encoder_try_new_validates_dt_seconds() {
        assert!(RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.001).is_ok());
        for dt in [0.0, -0.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), dt).is_err(),
                "dt_seconds={dt:?} should be rejected"
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_rate_encoder_serde_rejects_out_of_range_accumulators() {
        // Backlog whole spikes (>= 1.0) must round-trip after a capped step.
        let backlog = r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[5.0]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(backlog);
        assert!(res.is_ok());

        let negative =
            r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[-0.1]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(negative);
        assert!(res.is_err());

        let non_finite =
            r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[null]}"#;
        // JSON null is a type error
        let res: Result<RateEncoder, _> = serde_json::from_str(non_finite);
        assert!(res.is_err());

        let ok = r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"dt_seconds":0.1,"accumulators":[0.5]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(ok);
        assert!(res.is_ok());
    }

    #[test]
    fn test_rate_encoder_large_backlog_drains_exactly() {
        // Above ~2^24, f32 cannot subtract 1.0; u64 pending must still drain.
        let mut encoder = RateEncoder::try_new(0.0, 20_000_000.0, (0.0, 1.0), 1.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Quiet step must continue draining the same cap amount.
        let second = encoder.encode_step(&[0.0]);
        assert_eq!(
            second.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // After many drain steps, total emitted should exceed a single cap.
        let mut total = first.spikes.len() + second.spikes.len();
        for _ in 0..10 {
            total += encoder.encode_step(&[0.0]).spikes.len();
        }
        assert!(
            total > RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP * 2,
            "backlog should keep draining across steps, total={total}"
        );
    }

    #[test]
    fn test_rate_encoder_backlog_drains_above_f64_precision() {
        // Above 2^53, f64 `acc -= 1.0` is a no-op. Exact u64 pending must still
        // decrease so quiet steps eventually exhaust the queue instead of
        // emitting the 1024-spike cap forever.
        //
        // Seed a modest backlog via serde (above one cap, well below 2^53 so the
        // test finishes quickly) and a huge runtime increment past 2^53.
        #[cfg(feature = "serde")]
        {
            let seeded: RateEncoder = serde_json::from_str(
                r#"{"base_rate":0.0,"max_rate":1.0,"range":[0.0,1.0],"dt_seconds":1.0,"accumulators":[2500.0]}"#,
            )
            .unwrap();
            let mut encoder = seeded;
            let first = encoder.encode_step(&[0.0]);
            assert_eq!(
                first.spikes.len(),
                RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
            );
            let second = encoder.encode_step(&[0.0]);
            assert_eq!(
                second.spikes.len(),
                RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
            );
            // 2500 - 2*1024 = 452 remaining
            let third = encoder.encode_step(&[0.0]);
            assert_eq!(third.spikes.len(), 452);
            let fourth = encoder.encode_step(&[0.0]);
            assert!(
                fourth.spikes.is_empty(),
                "backlog must fully drain rather than emit forever"
            );
        }

        // Runtime path: 1e16 Hz × 1 s is past f64's exact integer range.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e16, (0.0, 1.0), 1.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Pending must decrease exactly by the cap each quiet step (not stall).
        let before = encoder.pending_spikes[0];
        assert!(
            before > RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64,
            "expected a large exact backlog, got {before}"
        );
        let quiet = encoder.encode_step(&[0.0]);
        assert_eq!(
            quiet.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        assert_eq!(
            encoder.pending_spikes[0],
            before - RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64,
            "u64 pending must decrement exactly past the f64 precision cliff"
        );
    }

    #[test]
    fn test_rate_encoder_streaming_bounds_extreme_dt() {
        // f32::MAX is a valid finite dt; f32 rate*dt would be +inf and was
        // previously dropped. f64 product stays finite and enqueues under the
        // per-step cap (must terminate, no OOM / hang).
        let mut encoder = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), f32::MAX).unwrap();
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(
            output.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );

        // Huge but finite expected count is capped per step; remainder is queued.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).unwrap();
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(
            output.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Undispatched whole spikes remain and drain on later quiet steps.
        let next = encoder.encode_step(&[0.0]);
        assert_eq!(
            next.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
    }

    #[test]
    fn test_rate_encoder_nan_input_is_silent() {
        let mut encoder = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.1).unwrap();
        let batch = encoder.encode(&[f32::NAN]);
        assert!(
            batch.spikes.is_empty(),
            "NaN sensor values must not map to max-rate / p≈1"
        );
        let step = encoder.encode_step(&[f32::NAN]);
        assert!(step.spikes.is_empty());
        assert_eq!(
            encoder.pending_spikes.first().copied().unwrap_or(0),
            0,
            "NaN must not seed a huge pending backlog"
        );
        // Finite input after NaN still works (state not poisoned).
        let ok = encoder.encode_step(&[1.0]);
        assert_eq!(ok.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_rate_dt_product_saturates_not_silent() {
        // max_rate ≈ 1e38, dt=10: f32 product overflows to +inf; f64 product
        // remains finite and must enqueue under the cap.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e38, (0.0, 1.0), 10.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP,
            "rate×dt f32 overflow must not silence streaming"
        );
        assert!(
            encoder.pending_spikes[0] > 0,
            "expected a queued backlog after the per-step cap"
        );
    }

    #[test]
    fn test_rate_encoder_inactive_gain_clears_pending() {
        let mut encoder = RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        assert!(encoder.pending_spikes[0] > 0);
        // Zero gain = documented silence: no spikes, backlog discarded.
        let silenced = encoder.encode_step_with_rate_scale(&[0.0], 0.0);
        assert!(silenced.spikes.is_empty());
        assert_eq!(encoder.pending_spikes[0], 0);
        assert_eq!(encoder.phases[0], 0.0);
        // After silence, a quiet identity step must not emit frozen backlog.
        let quiet = encoder.encode_step(&[0.0]);
        assert!(quiet.spikes.is_empty());
    }

    #[test]
    fn test_rate_encoder_default_dt_preserves_streaming_compatibility() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        assert_eq!(encoder.dt_seconds(), RateEncoder::DEFAULT_DT_SECONDS);
        assert_eq!(encoder.encode_step(&[1.0]).spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_overflowed_gain_rate_saturates_not_silent() {
        // max_rate * MAX_GAIN_SCALE (1e4) overflows f32 multiplication to +inf
        // when max_rate is ~1e35. Saturated effective rate must yield p ≈ 1, not
        // the non-finite-rate silence path in probability_from_rate_hz.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e35, (0.0, 1.0), 0.01).unwrap();
        let gains = EncodingGains {
            firing_rate_scale: 1.0e4,
            ..Default::default()
        };
        let mut spikes = 0usize;
        for _ in 0..32 {
            spikes += encoder.encode_with_gains(&[1.0], gains).spikes.len();
        }
        assert!(
            spikes >= 28,
            "overflowed modulated rate should saturate near p=1, got {spikes}/32 spikes"
        );
    }

    #[test]
    fn test_rate_encoder_streaming_uses_hz_times_dt() {
        let cases = [(5.0, 0.2, 10), (20.0, 0.05, 20), (7.5, 0.1, 40)];
        for (rate_hz, dt_seconds, steps) in cases {
            let mut encoder = RateEncoder::try_new(0.0, rate_hz, (0.0, 1.0), dt_seconds).unwrap();
            let spikes: usize = (0..steps)
                .map(|_| encoder.encode_step(&[1.0]).spikes.len())
                .sum();
            let elapsed_seconds = dt_seconds * steps as f32;
            let observed_hz = spikes as f32 / elapsed_seconds;
            assert!(
                (observed_hz - rate_hz).abs() <= 1.0 / elapsed_seconds,
                "rate_hz={rate_hz}, dt={dt_seconds}, observed={observed_hz}"
            );
        }
    }

    #[test]
    fn test_rate_encoder_stochastic_mean_matches_poisson_probability() {
        let cases = [(2.0, 0.01), (10.0, 0.005), (25.0, 0.002)];
        let trials = 50_000;
        for (rate_hz, dt_seconds) in cases {
            let mut encoder = RateEncoder::try_new(0.0, rate_hz, (0.0, 1.0), dt_seconds).unwrap();
            let spikes: usize = (0..trials)
                .map(|_| encoder.encode(&[1.0]).spikes.len())
                .sum();
            let observed_probability = spikes as f32 / trials as f32;
            let expected_probability =
                crate::poisson::probability_from_rate_hz(rate_hz, dt_seconds);
            assert!(
                (observed_probability - expected_probability).abs() < 0.01,
                "rate_hz={rate_hz}, dt={dt_seconds}, observed_p={observed_probability}, expected_p={expected_probability}"
            );
        }
    }
}

/// Property-style suites for rate / silence / bound contracts (#69 / LIM-1016).
///
/// Uses a seeded [`StdRng`] to sample configurations and inputs so failures are
/// reproducible. Encoder-internal spike draws still use the process RNG; the
/// invariants asserted here are independent of those draws.
#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::encoders::property_support::{
        TRIALS, assert_unique_channel_spikes, sample_gain_scale, sample_input_value,
        sample_positive_finite, scale_is_inactive,
    };
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Fixed seed so CI and local failures replay identically.
    const SEED: u64 = 0xAE69_0001;

    fn sample_valid_encoder(rng: &mut StdRng) -> RateEncoder {
        loop {
            let base = sample_positive_finite(rng) * rng.random::<f32>();
            let max = base + sample_positive_finite(rng);
            let lo = rng.random_range(-50.0_f32..50.0);
            let hi = lo + sample_positive_finite(rng);
            let dt = sample_positive_finite(rng).clamp(1e-4, 1.0);
            if let Ok(enc) = RateEncoder::try_new(base, max, (lo, hi), dt) {
                return enc;
            }
        }
    }

    fn sample_input_vec(rng: &mut StdRng, n: usize, range: (f32, f32)) -> Vec<f32> {
        (0..n).map(|_| sample_input_value(rng, range)).collect()
    }

    fn assert_both_silent(trial: usize, batch: &EncodedOutput, step: &EncodedOutput, why: &str) {
        assert!(
            batch.spikes.is_empty() && step.spikes.is_empty(),
            "trial {trial}: {why}"
        );
    }

    fn assert_active_batch_bounds(trial: usize, batch: &EncodedOutput, n_channels: usize) {
        assert!(
            batch.spikes.len() <= n_channels,
            "trial {trial}: batch spikes {} > channels {n_channels}",
            batch.spikes.len()
        );
        assert_unique_channel_spikes(&batch.spikes, n_channels);
    }

    fn assert_active_step_bounds(trial: usize, step: &EncodedOutput, n_channels: usize) {
        let max_step = RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP.saturating_mul(n_channels);
        assert!(
            step.spikes.len() <= max_step,
            "trial {trial}: step spikes {} exceed bound {max_step}",
            step.spikes.len()
        );
        for spike in &step.spikes {
            assert!((spike.channel as usize) < n_channels);
            assert!(spike.polarity);
        }
    }

    #[test]
    fn prop_rate_silence_and_channel_bounds() {
        let mut rng = StdRng::seed_from_u64(SEED);
        for trial in 0..TRIALS {
            let mut encoder = sample_valid_encoder(&mut rng);
            let n = rng.random_range(0usize..=8);
            let input = sample_input_vec(&mut rng, n, (0.0, 1.0));
            let scale = sample_gain_scale(&mut rng);

            let batch = encoder.encode_with_rate_scale(&input, scale);
            let step = encoder.encode_step_with_rate_scale(&input, scale);

            if input.is_empty() {
                assert_both_silent(
                    trial,
                    &batch,
                    &step,
                    "empty input must silence batch and step",
                );
                continue;
            }
            if scale_is_inactive(scale) {
                assert_both_silent(
                    trial,
                    &batch,
                    &step,
                    &format!("inactive rate_scale={scale:?} must silence"),
                );
                continue;
            }

            assert_active_batch_bounds(trial, &batch, input.len());
            assert_active_step_bounds(trial, &step, input.len());
            if input.iter().all(|v| !v.is_finite()) {
                assert!(
                    batch.spikes.is_empty(),
                    "trial {trial}: all non-finite inputs must silence batch"
                );
            }
        }
    }

    #[test]
    fn prop_rate_probability_stays_in_unit_interval() {
        let mut rng = StdRng::seed_from_u64(SEED ^ 0xB0B5);
        for _ in 0..TRIALS {
            let rate = match rng.random_range(0u8..8) {
                0 => 0.0,
                1 => -1.0,
                2 => f32::NAN,
                3 => f32::INFINITY,
                4 => f32::MAX,
                _ => sample_positive_finite(&mut rng) * rng.random_range(0.0_f32..100.0),
            };
            let dt = match rng.random_range(0u8..6) {
                0 => 0.0,
                1 => -0.1,
                2 => f32::NAN,
                3 => f32::INFINITY,
                _ => sample_positive_finite(&mut rng).clamp(1e-6, 2.0),
            };
            let p = crate::poisson::probability_from_rate_hz(rate, dt);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "probability_from_rate_hz({rate}, {dt}) = {p}"
            );
        }
    }

    #[test]
    fn prop_rate_encode_never_panics_on_sampled_inputs() {
        let mut rng = StdRng::seed_from_u64(SEED ^ 0xBAD5);
        for _ in 0..TRIALS {
            let mut encoder = sample_valid_encoder(&mut rng);
            let n = rng.random_range(0usize..=16);
            let input = sample_input_vec(&mut rng, n, (-10.0, 10.0));
            let scale = sample_gain_scale(&mut rng);
            let _ = encoder.encode_with_rate_scale(&input, scale);
            let _ = encoder.encode_step_with_rate_scale(&input, scale);
            encoder.reset();
            let _ = encoder.encode(&input);
            let _ = encoder.encode_step(&input);
        }
    }
}
