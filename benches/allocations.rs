use axon_encoder::encoders::{
    DeltaEncoder, LatencyEncoder, PopulationEncoder, PredictiveEncoder, RateEncoder,
    TemporalEncoder,
};
use axon_encoder::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SCALES: [usize; 3] = [256, 1024, 10_000];
const POISSON_STEPS: [usize; 3] = [10, 100, 1000];

struct CountingAllocator;

static COUNTING_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let ptr = unsafe { System.alloc(layout) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !ptr.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !ptr.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !new_ptr.is_null() {
            // Only count the net growth to avoid double-counting existing allocations.
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::SeqCst);
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // dealloc is not tracked to focus on net growth metrics during the measured operation.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Clone, Copy)]
struct AllocationStats {
    allocations: usize,
    bytes: usize,
    /// Spikes the measured step emitted.
    ///
    /// Reported so a row cannot claim "0 allocations" for a step that in fact
    /// emitted nothing: a zero here means the measurement is vacuous, not good.
    spikes: usize,
}

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

fn constant_input(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

fn measure_operation<T>(operation: impl FnOnce() -> T) -> AllocationStats {
    ALLOCATION_COUNT.store(0, Ordering::SeqCst);
    ALLOCATION_BYTES.store(0, Ordering::SeqCst);
    COUNTING_ENABLED.store(true, Ordering::SeqCst);
    let result = operation();
    COUNTING_ENABLED.store(false, Ordering::SeqCst);
    black_box(result);

    AllocationStats {
        allocations: ALLOCATION_COUNT.load(Ordering::SeqCst),
        bytes: ALLOCATION_BYTES.load(Ordering::SeqCst),
        spikes: 0,
    }
}

/// Measures a returning-path call and records how many spikes it produced.
fn measure_encode(operation: impl FnOnce() -> EncodedOutput) -> AllocationStats {
    let mut spikes = 0;
    let mut stats = measure_operation(|| {
        let output = operation();
        spikes = output.spikes.len();
        output
    });
    stats.spikes = spikes;
    stats
}

fn print_stats(
    encoder: &str,
    operation: &str,
    scale_label: &str,
    scale: usize,
    stats: AllocationStats,
) {
    println!(
        "{encoder},{operation},{scale_label},{scale},{},{},{}",
        stats.allocations, stats.bytes, stats.spikes
    );
}

/// Measures one steady-state step of the reusable path.
///
/// The buffer is warmed first so its capacity is already paid for; each measured
/// call then clears it (capacity survives) and refills it. Anything counted here
/// is an allocation the sink path failed to avoid.
fn measure_reused_step(
    warmups: usize,
    buffer: &mut Vec<SpikeEvent>,
    mut step: impl FnMut(&mut Vec<SpikeEvent>),
) -> AllocationStats {
    for _ in 0..warmups.max(1) {
        buffer.clear();
        step(buffer);
    }

    let mut stats = measure_operation(|| {
        buffer.clear();
        step(buffer);
    });
    stats.spikes = buffer.len();
    stats
}

fn report_rate_encoder() {
    for scale in SCALES {
        let mut encoder =
            RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), RateEncoder::DEFAULT_DT_SECONDS)
                .expect("valid RateEncoder");
        let input = normalized_input(scale);
        encoder.encode_step(&input);

        let stats = measure_encode(|| encoder.encode_step(&input));
        print_stats("RateEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_population_encoder() {
    for neurons in SCALES {
        let mut encoder = PopulationEncoder::try_new(neurons, (50.0, 100.0), 10.0)
            .expect("valid PopulationEncoder");
        let input = [75.0_f32];

        let stats = measure_encode(|| encoder.encode(&input));
        print_stats("PopulationEncoder", "encode", "neurons", neurons, stats);
    }
}

fn report_delta_encoder() {
    for scale in SCALES {
        let mut encoder = DeltaEncoder::try_new(0.1, scale).expect("valid DeltaEncoder");
        let baseline = normalized_input(scale);
        let shifted = shifted_input(scale, 0.25);
        encoder.encode_step(&baseline);

        let stats = measure_encode(|| encoder.encode_step(&shifted));
        print_stats("DeltaEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_temporal_encoder() {
    for scale in SCALES {
        let mut encoder =
            TemporalEncoder::try_new(6, vec![(0.2, 1)], scale).expect("valid TemporalEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);

        for input in [&low, &low, &low, &high, &high, &high] {
            encoder.encode_step(input);
        }

        let stats = measure_encode(|| encoder.encode_step(&high));
        print_stats("TemporalEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_predictive_encoder() {
    for scale in SCALES {
        let mut encoder =
            PredictiveEncoder::try_new(5, vec![(0.2, 1)], scale).expect("valid PredictiveEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);

        for _ in 0..5 {
            encoder.encode_step(&low);
        }

        let stats = measure_encode(|| encoder.encode_step(&high));
        print_stats("PredictiveEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_latency_encoder() {
    for scale in SCALES {
        let mut encoder = LatencyEncoder::try_new(15, (0.0, 1.0)).expect("valid LatencyEncoder");
        let input = normalized_input(scale);
        encoder.encode_step(&input);

        let stats = measure_encode(|| encoder.encode_step(&input));
        print_stats("LatencyEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_rate_encoder_into() {
    for scale in SCALES {
        let mut encoder =
            RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), RateEncoder::DEFAULT_DT_SECONDS)
                .expect("valid RateEncoder");
        let input = normalized_input(scale);
        // Pre-sized like a real caller would: a stochastic spike count must not
        // grow the buffer inside the measured step.
        let mut buffer = Vec::with_capacity(scale);

        let stats = measure_reused_step(2, &mut buffer, |sink| {
            encoder.encode_step_into(&input, sink);
        });
        print_stats("RateEncoder", "encode_step_into", "scale", scale, stats);
    }
}

fn report_population_encoder_into() {
    for neurons in SCALES {
        let mut encoder = PopulationEncoder::try_new(neurons, (50.0, 100.0), 10.0)
            .expect("valid PopulationEncoder");
        let input = [75.0_f32];
        // Up to `neurons` spikes can fire, and the count varies per call.
        let mut buffer = Vec::with_capacity(neurons);

        let stats = measure_reused_step(2, &mut buffer, |sink| {
            encoder.encode_into(&input, sink);
        });
        print_stats(
            "PopulationEncoder",
            "encode_into",
            "neurons",
            neurons,
            stats,
        );
    }
}

fn report_delta_encoder_into() {
    for scale in SCALES {
        let mut encoder = DeltaEncoder::try_new(0.1, scale).expect("valid DeltaEncoder");
        let baseline = normalized_input(scale);
        let shifted = shifted_input(scale, 0.25);
        let mut use_shifted = true;
        let mut buffer = Vec::with_capacity(scale);

        // Alternating inputs keep every channel crossing the threshold, so the
        // measured step emits a full-width burst rather than settling silent.
        let stats = measure_reused_step(2, &mut buffer, |sink| {
            let input = if use_shifted { &shifted } else { &baseline };
            use_shifted = !use_shifted;
            encoder.encode_step_into(input, sink);
        });
        print_stats("DeltaEncoder", "encode_step_into", "scale", scale, stats);
    }
}

fn report_temporal_encoder_into() {
    for scale in SCALES {
        let mut encoder =
            TemporalEncoder::try_new(6, vec![(0.2, 1)], scale).expect("valid TemporalEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);
        let mut buffer = Vec::with_capacity(scale);

        for input in [&low, &low, &low, &high, &high, &high] {
            encoder.encode_step(input);
        }

        // Alternate levels: a repeated input drives the history to a steady
        // state where the recent and older averages match, so nothing fires and
        // the measurement would be vacuous.
        let mut use_high = true;
        let stats = measure_reused_step(2, &mut buffer, |sink| {
            let input = if use_high { &high } else { &low };
            use_high = !use_high;
            encoder.encode_step_into(input, sink);
        });
        print_stats("TemporalEncoder", "encode_step_into", "scale", scale, stats);
    }
}

fn report_predictive_encoder_into() {
    for scale in SCALES {
        let mut encoder =
            PredictiveEncoder::try_new(5, vec![(0.2, 1)], scale).expect("valid PredictiveEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);
        let mut buffer = Vec::with_capacity(scale);

        for _ in 0..5 {
            encoder.encode_step(&low);
        }

        // Alternate levels so the prediction keeps being wrong; holding one
        // level lets the threshold converge and the measured step fall silent.
        let mut use_high = true;
        let stats = measure_reused_step(2, &mut buffer, |sink| {
            let input = if use_high { &high } else { &low };
            use_high = !use_high;
            encoder.encode_step_into(input, sink);
        });
        print_stats(
            "PredictiveEncoder",
            "encode_step_into",
            "scale",
            scale,
            stats,
        );
    }
}

fn report_latency_encoder_into() {
    for scale in SCALES {
        let mut encoder = LatencyEncoder::try_new(15, (0.0, 1.0)).expect("valid LatencyEncoder");
        let input = normalized_input(scale);
        let mut buffer = Vec::with_capacity(scale);

        let stats = measure_reused_step(2, &mut buffer, |sink| {
            encoder.encode_step_into(&input, sink);
        });
        print_stats("LatencyEncoder", "encode_step_into", "scale", scale, stats);
    }
}

/// Backlog drain on the returning path.
///
/// A rate high enough to queue more than `MAX_SPIKES_PER_CHANNEL_PER_STEP`
/// makes one channel emit a long run of coincident spikes in a single step.
/// That run is the case where a capacity hint earns its keep, and no other row
/// exercises it.
fn report_rate_encoder_backlog() {
    // The drain is per channel and always hits the same cap, so this does not
    // vary with the channel-count scales; report it once at its real width.
    const CHANNELS: usize = 8;

    let mut encoder =
        RateEncoder::try_new(0.0, 100_000.0, (0.0, 1.0), 0.1).expect("valid RateEncoder");
    let input = constant_input(CHANNELS, 1.0);
    encoder.encode_step(&input);

    let stats = measure_encode(|| encoder.encode_step(&input));
    print_stats(
        "RateEncoder",
        "encode_step(backlog)",
        "channels",
        CHANNELS,
        stats,
    );
}

fn report_poisson_encoder() {
    for steps in POISSON_STEPS {
        let encoder = PoissonEncoder::new(steps);
        let mut fired = 0;
        let mut stats = measure_operation(|| {
            let train = encoder.encode(0.5);
            fired = train.iter().filter(|&&bit| bit == 1).count();
            train
        });
        stats.spikes = fired;
        print_stats("PoissonEncoder", "encode", "steps", steps, stats);
    }
}

fn main() {
    println!("encoder,operation,scale_type,scale,allocations,bytes,spikes");
    report_rate_encoder();
    report_population_encoder();
    report_delta_encoder();
    report_temporal_encoder();
    report_predictive_encoder();
    report_latency_encoder();
    report_rate_encoder_backlog();
    // Reusable-storage counterparts: same encoders, same scales, one buffer.
    report_rate_encoder_into();
    report_population_encoder_into();
    report_delta_encoder_into();
    report_temporal_encoder_into();
    report_predictive_encoder_into();
    report_latency_encoder_into();
    report_poisson_encoder();
}
