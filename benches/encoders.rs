use axon_encoder::encoders::{
    DeltaEncoder, LatencyEncoder, PopulationEncoder, PredictiveEncoder, RateEncoder,
    TemporalEncoder,
};
use axon_encoder::prelude::*;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const SCALES: [usize; 3] = [256, 1024, 10_000];
const POISSON_STEPS: [usize; 3] = [10, 100, 1000];

fn normalized_input(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| i as f32 / (size.saturating_sub(1).max(1) as f32))
        .collect()
}

fn shifted_input(size: usize, offset: f32) -> Vec<f32> {
    normalized_input(size)
        .into_iter()
        .map(|value| (value + offset).clamp(0.0, 10.0))
        .collect()
}

fn temporal_level(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

fn bench_rate_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("RateEncoder::encode");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), RateEncoder::DEFAULT_DT_SECONDS)
                    .expect("valid RateEncoder");
            let input = normalized_input(size);
            b.iter(|| black_box(encoder.encode(black_box(&input))));
        });
    }
    group.finish();
}

fn bench_rate_encoder_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("RateEncoder::encode_step");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), RateEncoder::DEFAULT_DT_SECONDS)
                    .expect("valid RateEncoder");
            let input = normalized_input(size);
            encoder.encode_step(&input);

            b.iter(|| black_box(encoder.encode_step(black_box(&input))));
        });
    }
    group.finish();
}

fn bench_population_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PopulationEncoder::encode");
    for neurons in SCALES {
        group.bench_with_input(
            BenchmarkId::from_parameter(neurons),
            &neurons,
            |b, &neurons| {
                let mut encoder = PopulationEncoder::try_new(neurons, (50.0, 100.0), 10.0)
                    .expect("valid PopulationEncoder");
                let input = [75.0_f32];
                b.iter(|| black_box(encoder.encode(black_box(&input))));
            },
        );
    }
    group.finish();
}

fn bench_delta_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("DeltaEncoder::encode");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = DeltaEncoder::try_new(0.1, size).expect("valid DeltaEncoder");
            let input = shifted_input(size, 0.25);
            b.iter(|| black_box(encoder.encode(black_box(&input))));
        });
    }
    group.finish();
}

fn bench_delta_encoder_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("DeltaEncoder::encode_step");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = DeltaEncoder::try_new(0.1, size).expect("valid DeltaEncoder");
            let baseline = normalized_input(size);
            let shifted = shifted_input(size, 0.25);
            let mut use_shifted = true;

            encoder.encode_step(&baseline);

            b.iter(|| {
                let input = if use_shifted { &shifted } else { &baseline };
                use_shifted = !use_shifted;
                black_box(encoder.encode_step(black_box(input)))
            });
        });
    }
    group.finish();
}

fn bench_temporal_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("TemporalEncoder::encode");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                TemporalEncoder::try_new(6, vec![(0.2, 1)], size).expect("valid TemporalEncoder");
            let input = normalized_input(size);
            b.iter(|| black_box(encoder.encode(black_box(&input))));
        });
    }
    group.finish();
}

fn bench_temporal_encoder_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("TemporalEncoder::encode_step");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                TemporalEncoder::try_new(6, vec![(0.2, 1)], size).expect("valid TemporalEncoder");
            let low = temporal_level(size, 0.0);
            let high = temporal_level(size, 1.0);
            let sequence = [&low, &low, &low, &high, &high, &high];
            let mut index = 0usize;

            for input in sequence {
                encoder.encode_step(input);
            }
            // Cycle through inputs to ensure the encoder is tested under active temporal changes
            // rather than reaching a steady state with no spikes.
            b.iter(|| {
                let input = sequence[index % sequence.len()];
                index += 1;
                black_box(encoder.encode_step(black_box(input)))
            });
        });
    }
    group.finish();
}

fn bench_predictive_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PredictiveEncoder::encode");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = PredictiveEncoder::try_new(5, vec![(0.2, 1)], size)
                .expect("valid PredictiveEncoder");
            // Cycle low/high so thresholds keep adapting (static input converges and under-measures).
            let low = temporal_level(size, 0.0);
            let high = temporal_level(size, 1.0);
            let sequence = [&low, &low, &low, &high, &high, &high];
            let mut index = 0usize;

            for _ in 0..5 {
                encoder.encode(&low);
            }

            b.iter(|| {
                let input = sequence[index % sequence.len()];
                index += 1;
                black_box(encoder.encode(black_box(input)))
            });
        });
    }
    group.finish();
}

fn bench_predictive_encoder_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("PredictiveEncoder::encode_step");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = PredictiveEncoder::try_new(5, vec![(0.2, 1)], size)
                .expect("valid PredictiveEncoder");
            let low = temporal_level(size, 0.0);
            let high = temporal_level(size, 1.0);
            let sequence = [&low, &low, &low, &high, &high, &high];
            let mut index = 0usize;

            for _ in 0..5 {
                encoder.encode_step(&low);
            }

            b.iter(|| {
                let input = sequence[index % sequence.len()];
                index += 1;
                black_box(encoder.encode_step(black_box(input)))
            });
        });
    }
    group.finish();
}

// --- Reusable-storage (`SpikeSink`) counterparts ------------------------------
//
// Each of these mirrors the returning benchmark directly above its group, with
// one difference: a single buffer, cleared and refilled, replaces the per-call
// `Vec<SpikeEvent>`. Compare the pairs to see what per-spike virtual dispatch
// costs against what the removed allocation saves.

fn bench_rate_encoder_step_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("RateEncoder::encode_step_into");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), RateEncoder::DEFAULT_DT_SECONDS)
                    .expect("valid RateEncoder");
            let input = normalized_input(size);
            let mut buffer = Vec::with_capacity(size);
            encoder.encode_step_into(&input, &mut buffer);

            b.iter(|| {
                buffer.clear();
                encoder.encode_step_into(black_box(&input), &mut buffer);
                black_box(buffer.len())
            });
        });
    }
    group.finish();
}

fn bench_population_encoder_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("PopulationEncoder::encode_into");
    for neurons in SCALES {
        group.bench_with_input(
            BenchmarkId::from_parameter(neurons),
            &neurons,
            |b, &neurons| {
                let mut encoder = PopulationEncoder::try_new(neurons, (50.0, 100.0), 10.0)
                    .expect("valid PopulationEncoder");
                let input = [75.0_f32];
                // Up to `neurons` spikes fire, and the count varies per call.
                let mut buffer = Vec::with_capacity(neurons);
                encoder.encode_into(&input, &mut buffer);

                b.iter(|| {
                    buffer.clear();
                    encoder.encode_into(black_box(&input), &mut buffer);
                    black_box(buffer.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_delta_encoder_step_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("DeltaEncoder::encode_step_into");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = DeltaEncoder::try_new(0.1, size).expect("valid DeltaEncoder");
            let baseline = normalized_input(size);
            let shifted = shifted_input(size, 0.25);
            let mut use_shifted = true;
            let mut buffer = Vec::with_capacity(size);

            encoder.encode_step_into(&baseline, &mut buffer);

            b.iter(|| {
                let input = if use_shifted { &shifted } else { &baseline };
                use_shifted = !use_shifted;
                buffer.clear();
                encoder.encode_step_into(black_box(input), &mut buffer);
                black_box(buffer.len())
            });
        });
    }
    group.finish();
}

fn bench_temporal_encoder_step_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("TemporalEncoder::encode_step_into");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                TemporalEncoder::try_new(6, vec![(0.2, 1)], size).expect("valid TemporalEncoder");
            let low = temporal_level(size, 0.0);
            let high = temporal_level(size, 1.0);
            let sequence = [&low, &low, &low, &high, &high, &high];
            let mut index = 0usize;
            let mut buffer = Vec::with_capacity(size);

            for input in sequence {
                encoder.encode_step_into(input, &mut buffer);
                buffer.clear();
            }

            b.iter(|| {
                let input = sequence[index % sequence.len()];
                index += 1;
                buffer.clear();
                encoder.encode_step_into(black_box(input), &mut buffer);
                black_box(buffer.len())
            });
        });
    }
    group.finish();
}

fn bench_predictive_encoder_step_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("PredictiveEncoder::encode_step_into");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder = PredictiveEncoder::try_new(5, vec![(0.2, 1)], size)
                .expect("valid PredictiveEncoder");
            let low = temporal_level(size, 0.0);
            let high = temporal_level(size, 1.0);
            let sequence = [&low, &low, &low, &high, &high, &high];
            let mut index = 0usize;
            let mut buffer = Vec::with_capacity(size);

            for _ in 0..5 {
                encoder.encode_step_into(&low, &mut buffer);
                buffer.clear();
            }

            b.iter(|| {
                let input = sequence[index % sequence.len()];
                index += 1;
                buffer.clear();
                encoder.encode_step_into(black_box(input), &mut buffer);
                black_box(buffer.len())
            });
        });
    }
    group.finish();
}

fn bench_latency_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("LatencyEncoder::encode");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                LatencyEncoder::try_new(15, (0.0, 1.0)).expect("valid LatencyEncoder");
            let input = normalized_input(size);
            b.iter(|| black_box(encoder.encode(black_box(&input))));
        });
    }
    group.finish();
}

fn bench_latency_encoder_into(c: &mut Criterion) {
    // The densest emitter in the crate — one spike per channel, every call — so
    // this is the harshest test of per-spike virtual dispatch through the sink.
    let mut group = c.benchmark_group("LatencyEncoder::encode_into");
    for size in SCALES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut encoder =
                LatencyEncoder::try_new(15, (0.0, 1.0)).expect("valid LatencyEncoder");
            let input = normalized_input(size);
            let mut buffer = Vec::with_capacity(size);
            encoder.encode_into(&input, &mut buffer);

            b.iter(|| {
                buffer.clear();
                encoder.encode_into(black_box(&input), &mut buffer);
                black_box(buffer.len())
            });
        });
    }
    group.finish();
}

fn bench_poisson_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PoissonEncoder::encode");
    for steps in POISSON_STEPS {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &steps| {
            let enc = PoissonEncoder::new(steps);
            b.iter(|| black_box(enc.encode(black_box(0.5))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rate_encoder,
    bench_rate_encoder_step,
    bench_population_encoder,
    bench_delta_encoder,
    bench_delta_encoder_step,
    bench_temporal_encoder,
    bench_temporal_encoder_step,
    bench_predictive_encoder,
    bench_predictive_encoder_step,
    bench_latency_encoder,
    bench_rate_encoder_step_into,
    bench_population_encoder_into,
    bench_delta_encoder_step_into,
    bench_temporal_encoder_step_into,
    bench_predictive_encoder_step_into,
    bench_latency_encoder_into,
    bench_poisson_encoder
);
criterion_main!(benches);
