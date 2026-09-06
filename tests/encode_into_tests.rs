//! Equivalence, reuse, and object-safety guarantees for the `SpikeSink` path.
//!
//! The sink-based API is only useful if it is a *drop-in* for the returning
//! API: same spikes, same order, same state advancement. These tests pin that
//! down encoder by encoder, plus the two properties the returning API cannot
//! offer — a buffer that survives across steps, and dispatch through
//! `&mut dyn Encoder` into `&mut dyn SpikeSink`.

use axon_encoder::prelude::*;

/// Encodes twice from identical starting states and compares the two paths.
///
/// Each closure gets its own encoder, so stateful encoders see the same history.
fn assert_paths_agree<E: Encoder>(
    label: &str,
    mut returning: E,
    mut sink_based: E,
    steps: &[&[f32]],
) {
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for (index, input) in steps.iter().enumerate() {
        let expected = returning.encode_step(input).spikes;

        buffer.clear();
        sink_based.encode_step_into(input, &mut buffer);

        assert_eq!(buffer, expected, "{label}: step {index} diverged");
    }
}

/// Batch counterpart of [`assert_paths_agree`]: `encode_into` against `encode`.
fn assert_batch_paths_agree<E: Encoder>(
    label: &str,
    mut returning: E,
    mut sink_based: E,
    steps: &[&[f32]],
) {
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for (index, input) in steps.iter().enumerate() {
        let expected = returning.encode(input).spikes;

        buffer.clear();
        sink_based.encode_into(input, &mut buffer);

        assert_eq!(buffer, expected, "{label}: batch step {index} diverged");
    }
}

/// Gain-scaled equivalence: `encode_with_gains_into` against `encode_with_gains`.
fn assert_gain_paths_agree<E: ModulatedEncoder>(
    label: &str,
    mut returning: E,
    mut sink_based: E,
    gains: EncodingGains,
    steps: &[&[f32]],
) {
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for (index, input) in steps.iter().enumerate() {
        let expected = returning.encode_with_gains(input, gains).spikes;

        buffer.clear();
        sink_based.encode_with_gains_into(input, gains, &mut buffer);

        assert_eq!(buffer, expected, "{label}: gain step {index} diverged");
    }
}

#[test]
fn delta_encoder_sink_path_matches_returning_path() {
    let steps: &[&[f32]] = &[&[0.0, 0.0], &[0.5, 0.0], &[0.5, -0.9], &[0.51, -0.9]];
    assert_paths_agree(
        "DeltaEncoder",
        DeltaEncoder::new(0.1, 2),
        DeltaEncoder::new(0.1, 2),
        steps,
    );
}

#[test]
fn derivative_encoder_sink_path_matches_returning_path() {
    let steps: &[&[f32]] = &[&[0.0, 0.0], &[1.0, -1.0], &[1.0, -1.0], &[-3.0, 4.0]];
    assert_paths_agree(
        "DerivativeEncoder",
        DerivativeEncoder::new(vec![0.5, 0.5]),
        DerivativeEncoder::new(vec![0.5, 0.5]),
        steps,
    );
}

#[test]
fn latency_encoder_sink_path_matches_returning_path() {
    let steps: &[&[f32]] = &[&[1.0, 0.0, 0.5], &[f32::NAN, 2.0, -1.0]];
    assert_paths_agree(
        "LatencyEncoder",
        LatencyEncoder::new(9, (0.0, 1.0)),
        LatencyEncoder::new(9, (0.0, 1.0)),
        steps,
    );
}

#[test]
fn phase_encoder_sink_path_matches_returning_path() {
    let steps: &[&[f32]] = &[&[0.0, 0.5, 1.0], &[1.0, f32::NAN, 0.25], &[0.75, 0.75, 0.0]];
    assert_paths_agree(
        "PhaseEncoder",
        PhaseEncoder::new(8, (0.0, 1.0)),
        PhaseEncoder::new(8, (0.0, 1.0)),
        steps,
    );
}

#[test]
fn phase_encoder_sink_path_advances_the_background_phase() {
    let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    assert_eq!(encoder.current_phase(), 0);
    encoder.encode_into(&[0.5], &mut buffer);
    assert_eq!(encoder.current_phase(), 1, "encode_into must advance phase");

    buffer.clear();
    encoder.encode_step_into(&[0.5], &mut buffer);
    assert_eq!(encoder.current_phase(), 2);
}

#[test]
fn predictive_encoder_sink_path_matches_returning_path() {
    // Five warm-up steps establish the prediction, then a jump forces spikes.
    let low: &[f32] = &[0.0, 0.0];
    let high: &[f32] = &[1.0, -1.0];
    let steps: &[&[f32]] = &[low, low, low, low, low, high, high, low];
    assert_paths_agree(
        "PredictiveEncoder",
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        steps,
    );
}

#[test]
fn temporal_encoder_sink_path_matches_returning_path() {
    let low: &[f32] = &[0.0, 0.0];
    let high: &[f32] = &[1.0, 1.0];
    let steps: &[&[f32]] = &[low, low, low, high, high, high, low, high];
    assert_paths_agree(
        "TemporalEncoder",
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        steps,
    );
}

#[test]
fn rate_encoder_streaming_sink_path_matches_returning_path() {
    // `encode_step` is deterministic (accumulator-driven), so this is exact.
    let steps: &[&[f32]] = &[&[1.0, 0.25], &[1.0, 0.25], &[1.0, 0.25], &[0.0, 1.0]];
    assert_paths_agree(
        "RateEncoder",
        RateEncoder::try_new(0.0, 30.0, (0.0, 1.0), 0.1).expect("valid RateEncoder"),
        RateEncoder::try_new(0.0, 30.0, (0.0, 1.0), 0.1).expect("valid RateEncoder"),
        steps,
    );
}

#[test]
fn rate_encoder_sink_path_drains_the_capped_backlog_identically() {
    // 100 kHz over a 100 ms step queues far more than the 1024/step cap, so the
    // backlog must drain across calls on both paths in lockstep.
    let mut returning = RateEncoder::try_new(0.0, 100_000.0, (0.0, 1.0), 0.1).expect("valid");
    let mut sink_based = RateEncoder::try_new(0.0, 100_000.0, (0.0, 1.0), 0.1).expect("valid");
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for step in 0..4 {
        let expected = returning.encode_step(&[1.0]).spikes;
        buffer.clear();
        sink_based.encode_step_into(&[1.0], &mut buffer);

        assert_eq!(expected.len(), 1024, "step {step} should hit the cap");
        assert_eq!(buffer, expected, "step {step} diverged");
    }
}

#[test]
fn rate_encoder_batch_sink_path_preserves_structural_invariants() {
    // Batch mode draws fresh entropy per call, so equality is not available;
    // assert the same invariants the returning path is held to.
    let mut encoder = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010).expect("valid");
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for _ in 0..32 {
        buffer.clear();
        encoder.encode_into(&input, &mut buffer);

        assert!(buffer.len() <= input.len(), "at most one spike per channel");
        let mut channels: Vec<u16> = buffer.iter().map(|spike| spike.channel).collect();
        channels.sort_unstable();
        let unique = channels.len();
        channels.dedup();
        assert_eq!(
            channels.len(),
            unique,
            "channels must not repeat in batch mode"
        );
        assert!(buffer.iter().all(|spike| spike.timestamp == 0));
        assert!(buffer.iter().all(|spike| spike.polarity));
    }
}

#[test]
fn rate_encoder_batch_sink_path_matches_returning_path_on_average() {
    // Statistical equivalence: identical configuration, so the expected spike
    // count per call must agree well within sampling noise over many trials.
    const TRIALS: usize = 400;
    let input: Vec<f32> = vec![0.5; 64];

    let mut returning = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010).expect("valid");
    let mut sink_based = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010).expect("valid");
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    let mut returning_total = 0usize;
    let mut sink_total = 0usize;
    for _ in 0..TRIALS {
        returning_total += returning.encode(&input).spikes.len();
        buffer.clear();
        sink_based.encode_into(&input, &mut buffer);
        sink_total += buffer.len();
    }

    let returning_mean = returning_total as f64 / TRIALS as f64;
    let sink_mean = sink_total as f64 / TRIALS as f64;
    assert!(
        (returning_mean - sink_mean).abs() < 1.0,
        "mean spikes per call diverged: returning {returning_mean}, sink {sink_mean}"
    );
}

#[test]
fn population_encoder_sink_path_preserves_structural_invariants() {
    let mut encoder = PopulationEncoder::new(64, (0.0, 100.0), 10.0);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for _ in 0..32 {
        buffer.clear();
        encoder.encode_into(&[50.0], &mut buffer);

        assert!(buffer.len() <= 64);
        let mut channels: Vec<u16> = buffer.iter().map(|spike| spike.channel).collect();
        assert!(channels.windows(2).all(|pair| pair[0] < pair[1]));
        channels.dedup();
        assert_eq!(channels.len(), buffer.len());
    }

    // Empty input silences both paths.
    buffer.clear();
    encoder.encode_into(&[], &mut buffer);
    assert!(buffer.is_empty());
    assert!(encoder.encode(&[]).spikes.is_empty());
}

#[test]
fn population_encoder_sink_path_matches_returning_path_on_average() {
    const TRIALS: usize = 400;
    let mut returning = PopulationEncoder::new(64, (0.0, 100.0), 10.0);
    let mut sink_based = PopulationEncoder::new(64, (0.0, 100.0), 10.0);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    let mut returning_total = 0usize;
    let mut sink_total = 0usize;
    for _ in 0..TRIALS {
        returning_total += returning.encode(&[50.0]).spikes.len();
        buffer.clear();
        sink_based.encode_into(&[50.0], &mut buffer);
        sink_total += buffer.len();
    }

    let returning_mean = returning_total as f64 / TRIALS as f64;
    let sink_mean = sink_total as f64 / TRIALS as f64;
    assert!(
        (returning_mean - sink_mean).abs() < 1.0,
        "mean spikes per call diverged: returning {returning_mean}, sink {sink_mean}"
    );
}

#[test]
fn batch_and_streaming_sink_entry_points_match_their_returning_twins() {
    // `encode_into` tracks `encode`, `encode_step_into` tracks `encode_step` —
    // the distinction matters for encoders whose streaming path differs.
    let mut returning = DeltaEncoder::new(0.1, 2);
    let mut sink_based = DeltaEncoder::new(0.1, 2);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    // Batch path, including the over-long input `encode` does not truncate.
    let expected = returning.encode(&[0.5, 0.9, 7.0]).spikes;
    sink_based.encode_into(&[0.5, 0.9, 7.0], &mut buffer);
    assert_eq!(buffer, expected);

    // Streaming path truncates to the configured channel count on both sides.
    let expected = returning.encode_step(&[2.0, 2.0, 9.0]).spikes;
    buffer.clear();
    sink_based.encode_step_into(&[2.0, 2.0, 9.0], &mut buffer);
    assert_eq!(buffer, expected);
}

#[test]
fn one_buffer_is_reused_across_streaming_steps_without_reallocating() {
    let channels = 512;
    let mut encoder = LatencyEncoder::new(15, (0.0, 1.0));
    let input: Vec<f32> = (0..channels)
        .map(|i| i as f32 / (channels - 1) as f32)
        .collect();

    let mut buffer: Vec<SpikeEvent> = Vec::new();
    encoder.encode_step_into(&input, &mut buffer); // first call sizes the buffer
    assert_eq!(buffer.len(), channels);
    let capacity_after_warmup = buffer.capacity();

    for step in 0..16 {
        buffer.clear();
        encoder.encode_step_into(&input, &mut buffer);

        assert_eq!(buffer.len(), channels, "step {step} lost spikes");
        assert_eq!(
            buffer.capacity(),
            capacity_after_warmup,
            "step {step} reallocated a warm buffer"
        );
    }
}

#[test]
fn a_sink_is_appended_to_never_cleared() {
    let mut encoder = LatencyEncoder::new(3, (0.0, 1.0));
    let mut buffer = vec![SpikeEvent::new(99, 42u64, false)];

    encoder.encode_into(&[1.0], &mut buffer);

    assert_eq!(buffer.len(), 2, "the pre-existing spike must survive");
    assert_eq!(buffer[0], SpikeEvent::new(99, 42u64, false));
    assert_eq!(buffer[1], SpikeEvent::new(0, 0u64, true));
}

#[test]
fn encoded_output_works_as_a_sink() {
    let mut encoder = LatencyEncoder::new(9, (0.0, 1.0));
    let mut output = EncodedOutput::new();

    encoder.encode_into(&[1.0, 0.0], &mut output);

    assert_eq!(output.spikes, encoder.encode(&[1.0, 0.0]).spikes);
    assert!(output.embeddings.is_none(), "sinks only carry spikes");
    assert!(output.metadata.is_none(), "sinks only carry spikes");
}

/// A caller-native buffer: proves an out-of-crate sink needs no `Vec<SpikeEvent>`.
#[derive(Default)]
struct EventLog {
    entries: Vec<(u16, u64, bool)>,
    reserved: usize,
}

impl SpikeSink for EventLog {
    fn push(&mut self, event: SpikeEvent) {
        self.entries
            .push((event.channel, event.timestamp.ticks(), event.polarity));
    }

    fn reserve(&mut self, additional: usize) {
        self.reserved += additional;
        self.entries.reserve(additional);
    }
}

fn encode_via_dyn_sink(encoder: &mut dyn Encoder, input: &[f32], sink: &mut dyn SpikeSink) {
    encoder.encode_into(input, sink);
}

#[test]
fn custom_sink_receives_spikes_through_dyn_dispatch() {
    let mut encoder = LatencyEncoder::new(9, (0.0, 1.0));
    let mut log = EventLog::default();

    encode_via_dyn_sink(&mut encoder, &[1.0, 0.0], &mut log);

    assert_eq!(log.entries, vec![(0, 0, true), (1, 9, true)]);
    assert!(log.reserved >= 2, "LatencyEncoder should pre-size its sink");
}

#[test]
fn modulated_encoders_reach_the_sink_through_dyn_dispatch() {
    fn encode_with_gains_via_dyn(
        encoder: &mut dyn ModulatedEncoder,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        encoder.encode_with_gains_into(input, gains, sink);
    }

    let gains = EncodingGains {
        latency_scale: 0.5,
        ..EncodingGains::identity()
    };

    let mut returning = LatencyEncoder::new(10, (0.0, 1.0));
    let expected = returning.encode_with_gains(&[0.0, 1.0], gains).spikes;

    let mut sink_based = LatencyEncoder::new(10, (0.0, 1.0));
    let mut buffer: Vec<SpikeEvent> = Vec::new();
    encode_with_gains_via_dyn(&mut sink_based, &[0.0, 1.0], gains, &mut buffer);

    assert_eq!(buffer, expected);
    // The gain really is applied: half the window, so the weak input lands at 5.
    assert_eq!(buffer[0].timestamp, 5);
}

#[test]
fn modulated_sink_path_matches_the_returning_modulator_path() {
    let modulators = NeuroModulators {
        dopamine: 1.0,
        ..Default::default()
    };
    let curves = NeuromodulatorGainCurves {
        dopamine: ModulatorGainCurves {
            firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut returning = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.1).expect("valid");
    let mut sink_based = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.1).expect("valid");
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    let mut total = 0usize;
    for step in 0..8 {
        let expected = returning
            .encode_step_with_modulators(&[1.0], &modulators, &curves)
            .spikes;

        buffer.clear();
        sink_based.encode_step_with_modulators_into(&[1.0], &modulators, &curves, &mut buffer);

        assert_eq!(buffer, expected, "modulated step {step} diverged");
        total += buffer.len();
    }

    // 10 Hz * 2x dopamine gain * 0.1 s = 2 spikes per step.
    assert_eq!(total, 16);
}

/// An out-of-crate encoder that does *not* override the sink methods.
struct PassThrough;

impl Encoder for PassThrough {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            if value > 0.0 {
                output
                    .spikes
                    .push(SpikeEvent::at_step_start(i as u16, true));
            }
        }
        output
    }

    fn reset(&mut self) {}
}

#[test]
fn inherited_sink_defaults_work_for_out_of_crate_encoders() {
    let mut encoder = PassThrough;
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    encoder.encode_into(&[1.0, 0.0, 2.0], &mut buffer);
    assert_eq!(
        buffer,
        vec![
            SpikeEvent::at_step_start(0, true),
            SpikeEvent::at_step_start(2, true),
        ]
    );

    // The streaming default follows `encode_step`, which itself defaults to `encode`.
    buffer.clear();
    encoder.encode_step_into(&[0.0, 3.0], &mut buffer);
    assert_eq!(buffer, vec![SpikeEvent::at_step_start(1, true)]);
}

// --- Batch entry point (`encode_into`) on every deterministic encoder ---------

#[test]
fn batch_sink_path_matches_encode_for_deterministic_encoders() {
    let two_channel: &[&[f32]] = &[&[0.0, 0.0], &[0.5, -0.9], &[0.51, -0.9], &[2.0, 2.0]];

    assert_batch_paths_agree(
        "DeltaEncoder",
        DeltaEncoder::new(0.1, 2),
        DeltaEncoder::new(0.1, 2),
        two_channel,
    );
    assert_batch_paths_agree(
        "DerivativeEncoder",
        DerivativeEncoder::new(vec![0.5, 0.5]),
        DerivativeEncoder::new(vec![0.5, 0.5]),
        two_channel,
    );
    assert_batch_paths_agree(
        "LatencyEncoder",
        LatencyEncoder::new(9, (0.0, 1.0)),
        LatencyEncoder::new(9, (0.0, 1.0)),
        &[&[1.0, 0.0, 0.5], &[f32::NAN, 2.0, -1.0]],
    );
    assert_batch_paths_agree(
        "PhaseEncoder",
        PhaseEncoder::new(8, (0.0, 1.0)),
        PhaseEncoder::new(8, (0.0, 1.0)),
        &[&[0.0, 0.5, 1.0], &[1.0, f32::NAN, 0.25], &[0.75, 0.75, 0.0]],
    );

    // Five warm-up steps establish the prediction, then a jump forces spikes.
    let low: &[f32] = &[0.0, 0.0];
    let high: &[f32] = &[1.0, -1.0];
    assert_batch_paths_agree(
        "PredictiveEncoder",
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        &[low, low, low, low, low, high, high, low],
    );
    assert_batch_paths_agree(
        "TemporalEncoder",
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        &[low, low, low, high, high, high, low, high],
    );
}

#[test]
fn population_encoder_streaming_sink_entry_point_holds_the_invariants() {
    // `encode_step_into` is its own entry point even where streaming delegates
    // to the batch path, so it needs its own exercise.
    let mut encoder = PopulationEncoder::new(32, (0.0, 100.0), 10.0);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for _ in 0..16 {
        buffer.clear();
        encoder.encode_step_into(&[50.0], &mut buffer);

        assert!(buffer.len() <= 32);
        assert!(buffer.iter().all(|spike| spike.timestamp == 0));
    }
}

// --- Gain-scaled sink path ----------------------------------------------------

#[test]
fn gain_sink_path_matches_returning_path_for_deterministic_encoders() {
    let threshold_gains = EncodingGains {
        threshold_scale: 2.0,
        ..EncodingGains::identity()
    };
    let two_channel: &[&[f32]] = &[&[0.0, 0.0], &[0.5, -0.9], &[0.51, -0.9], &[2.0, 2.0]];

    assert_gain_paths_agree(
        "DeltaEncoder",
        DeltaEncoder::new(0.1, 2),
        DeltaEncoder::new(0.1, 2),
        threshold_gains,
        two_channel,
    );

    let low: &[f32] = &[0.0, 0.0];
    let high: &[f32] = &[1.0, -1.0];
    assert_gain_paths_agree(
        "PredictiveEncoder",
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        PredictiveEncoder::try_new(5, vec![(0.2, 1)], 2).expect("valid PredictiveEncoder"),
        threshold_gains,
        &[low, low, low, low, low, high, high, low],
    );
    assert_gain_paths_agree(
        "TemporalEncoder",
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        TemporalEncoder::try_new(6, vec![(0.2, 1)], 2).expect("valid TemporalEncoder"),
        threshold_gains,
        &[low, low, low, high, high, high, low, high],
    );

    let sensitivity_gains = EncodingGains {
        sensitivity_scale: 0.5,
        ..EncodingGains::identity()
    };
    assert_gain_paths_agree(
        "PhaseEncoder",
        PhaseEncoder::new(8, (0.0, 1.0)),
        PhaseEncoder::new(8, (0.0, 1.0)),
        sensitivity_gains,
        &[&[0.0, 0.5, 1.0], &[1.0, f32::NAN, 0.25], &[0.75, 0.75, 0.0]],
    );
}

#[test]
fn gain_sink_path_advances_phase_exactly_like_the_returning_path() {
    let gains = EncodingGains {
        sensitivity_scale: 0.5,
        ..EncodingGains::identity()
    };
    let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    encoder.encode_with_gains_into(&[0.5], gains, &mut buffer);
    assert_eq!(encoder.current_phase(), 1);
}

#[test]
fn stochastic_gain_sink_paths_hold_their_invariants() {
    // Batch draws fresh entropy, so assert bounds and the documented silencing
    // behavior rather than equality.
    let mut rate = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010).expect("valid");
    let mut population = PopulationEncoder::new(32, (0.0, 100.0), 10.0);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    let boosted = EncodingGains {
        firing_rate_scale: 2.0,
        sensitivity_scale: 2.0,
        ..EncodingGains::identity()
    };
    for _ in 0..16 {
        buffer.clear();
        rate.encode_with_gains_into(&[0.5; 16], boosted, &mut buffer);
        assert!(buffer.len() <= 16);

        buffer.clear();
        population.encode_with_gains_into(&[50.0], boosted, &mut buffer);
        assert!(buffer.len() <= 32);
    }

    // A zero scale silences both, on the sink path as on the returning one.
    let silent = EncodingGains {
        firing_rate_scale: 0.0,
        sensitivity_scale: 0.0,
        ..EncodingGains::identity()
    };
    buffer.clear();
    rate.encode_with_gains_into(&[1.0; 16], silent, &mut buffer);
    assert!(buffer.is_empty());
    buffer.clear();
    population.encode_with_gains_into(&[50.0], silent, &mut buffer);
    assert!(buffer.is_empty());
}

#[test]
fn batch_modulator_sink_path_matches_the_returning_path() {
    let modulators = NeuroModulators {
        acetylcholine: 1.0,
        ..Default::default()
    };
    let curves = NeuromodulatorGainCurves {
        acetylcholine: ModulatorGainCurves {
            threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 3.0))),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut returning = DeltaEncoder::new(0.1, 3);
    let mut sink_based = DeltaEncoder::new(0.1, 3);
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for (index, input) in [[0.2_f32, 0.05, 0.5], [0.2, 0.05, 1.0]].iter().enumerate() {
        let expected = returning
            .encode_with_modulators(input, &modulators, &curves)
            .spikes;

        buffer.clear();
        sink_based.encode_with_modulators_into(input, &modulators, &curves, &mut buffer);

        assert_eq!(buffer, expected, "modulated batch step {index} diverged");
    }
}

/// An out-of-crate modulated encoder that overrides no sink method.
///
/// Every `*_into` call on it therefore runs the inherited defaults on
/// `Encoder` and `ModulatedEncoder`.
struct ThresholdToy {
    threshold: f32,
}

impl Encoder for ThresholdToy {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_with_gains(self, input, EncodingGains::identity())
    }

    fn reset(&mut self) {}
}

impl ModulatedEncoder for ThresholdToy {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let threshold = self.threshold * gains.sanitize().threshold_scale;
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            if value > threshold {
                output
                    .spikes
                    .push(SpikeEvent::at_step_start(i as u16, true));
            }
        }
        output
    }
}

#[test]
fn inherited_modulated_sink_defaults_work_for_out_of_crate_encoders() {
    let doubled = EncodingGains {
        threshold_scale: 2.0,
        ..EncodingGains::identity()
    };
    let input = [0.4_f32, 0.9, 1.5];
    let mut encoder = ThresholdToy { threshold: 0.5 };
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    // Gain defaults: threshold 0.5 * 2.0 = 1.0, so only channel 2 clears it.
    encoder.encode_with_gains_into(&input, doubled, &mut buffer);
    assert_eq!(buffer, vec![SpikeEvent::at_step_start(2, true)]);

    buffer.clear();
    encoder.encode_step_with_gains_into(&input, doubled, &mut buffer);
    assert_eq!(buffer, vec![SpikeEvent::at_step_start(2, true)]);

    // Modulator defaults: identity curves leave the threshold at 0.5.
    let modulators = NeuroModulators::default();
    let curves = NeuromodulatorGainCurves::default();
    let unscaled = vec![
        SpikeEvent::at_step_start(1, true),
        SpikeEvent::at_step_start(2, true),
    ];

    buffer.clear();
    encoder.encode_with_modulators_into(&input, &modulators, &curves, &mut buffer);
    assert_eq!(buffer, unscaled);

    buffer.clear();
    encoder.encode_step_with_modulators_into(&input, &modulators, &curves, &mut buffer);
    assert_eq!(buffer, unscaled);
}

// --- Chunked flushing --------------------------------------------------------

#[test]
fn chunked_flushing_delivers_every_spike_in_order() {
    // The sink path buffers spikes internally and flushes them in runs. A spike
    // count that straddles several run boundaries — and is not a multiple of
    // one — must still arrive complete and in the returning path's order.
    let channels = 205;
    let mut encoder = LatencyEncoder::new(15, (0.0, 1.0));
    let input: Vec<f32> = (0..channels)
        .map(|i| i as f32 / (channels - 1) as f32)
        .collect();

    let mut buffer: Vec<SpikeEvent> = Vec::new();
    encoder.encode_into(&input, &mut buffer);

    assert_eq!(buffer.len(), channels);
    assert_eq!(buffer, encoder.encode(&input).spikes);
    assert!(
        buffer
            .iter()
            .enumerate()
            .all(|(i, spike)| spike.channel == i as u16),
        "channels must stay in ascending order across flushes"
    );
}

/// A sink that implements only `push`, inheriting the default `extend_from_slice`.
#[derive(Default)]
struct PushOnlySink(Vec<SpikeEvent>);

impl SpikeSink for PushOnlySink {
    fn push(&mut self, event: SpikeEvent) {
        self.0.push(event);
    }
}

/// A sink that overrides `extend_from_slice` and records the run lengths it saw.
#[derive(Default)]
struct RunRecordingSink {
    spikes: Vec<SpikeEvent>,
    runs: Vec<usize>,
}

impl SpikeSink for RunRecordingSink {
    fn push(&mut self, event: SpikeEvent) {
        self.spikes.push(event);
        self.runs.push(1);
    }

    fn extend_from_slice(&mut self, events: &[SpikeEvent]) {
        self.spikes.extend_from_slice(events);
        self.runs.push(events.len());
    }
}

#[test]
fn custom_sinks_see_every_spike_whether_or_not_they_batch() {
    let channels = 205;
    let input: Vec<f32> = (0..channels)
        .map(|i| i as f32 / (channels - 1) as f32)
        .collect();

    let expected = LatencyEncoder::new(15, (0.0, 1.0)).encode(&input).spikes;

    let mut push_only = PushOnlySink::default();
    LatencyEncoder::new(15, (0.0, 1.0)).encode_into(&input, &mut push_only);
    assert_eq!(push_only.0, expected, "default extend_from_slice must agree");

    let mut recording = RunRecordingSink::default();
    LatencyEncoder::new(15, (0.0, 1.0)).encode_into(&input, &mut recording);
    assert_eq!(recording.spikes, expected);
    assert_eq!(
        recording.runs.iter().sum::<usize>(),
        channels,
        "the recorded runs must account for every spike"
    );
    assert!(
        recording.runs.len() < channels,
        "spikes should arrive batched, not one call per spike"
    );
}
