//! Encode-Into / SpikeSink Example
//!
//! Shows the allocation-reusing integration path: instead of taking the
//! `Vec<SpikeEvent>` that `encode_step` allocates on every call, a downstream
//! runtime implements `SpikeSink` and receives spikes in its own representation.

use axon_encoder::prelude::*;

/// A downstream runtime's event buffer.
///
/// Stores spikes as packed `(channel, absolute_tick)` pairs — the form the
/// imaginary consumer actually wants — so no `Vec<SpikeEvent>` is ever built.
/// The backing storage is allocated once and refilled every step.
struct EventBuffer {
    events: Vec<(u16, u64)>,
    /// Absolute tick of the current step, kept by the caller (this crate owns
    /// no clock: spike timestamps are offsets from the start of the call).
    origin: u64,
}

impl EventBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            origin: 0,
        }
    }

    /// Drops the previous step's events, keeping the allocated capacity.
    fn begin_step(&mut self, origin: u64) {
        self.events.clear();
        self.origin = origin;
    }
}

impl SpikeSink for EventBuffer {
    fn push(&mut self, event: SpikeEvent) {
        // Resolve the call-relative offset against the caller's timeline.
        self.events
            .push((event.channel, self.origin + event.timestamp.ticks()));
    }

    fn reserve(&mut self, additional: usize) {
        self.events.reserve(additional);
    }
}

fn main() {
    println!("=== Encode-Into / SpikeSink Example ===");

    let channels = 8;
    let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0)).expect("valid LatencyEncoder");
    let mut cursor = TimeCursor::new(encoder.time_model());
    let mut buffer = EventBuffer::with_capacity(channels);

    println!(
        "Encoder: LatencyEncoder over {channels} channels, window {} ticks",
        encoder.time_model().span_ticks()
    );
    let capacity_at_start = buffer.events.capacity();
    println!("Buffer capacity before encoding: {capacity_at_start}\n");

    for step in 0..5 {
        // Signal sweeps upward each step; stronger input spikes earlier.
        let inputs: Vec<f32> = (0..channels)
            .map(|i| ((i + step) % channels) as f32 / (channels - 1) as f32)
            .collect();

        buffer.begin_step(cursor.origin());
        encoder.encode_step_into(&inputs, &mut buffer);
        cursor.advance();

        let earliest = buffer.events.iter().map(|(_, tick)| *tick).min();
        println!(
            "Step {step}: {} events, origin {}, earliest absolute tick {:?}",
            buffer.events.len(),
            buffer.origin,
            earliest
        );
    }

    println!(
        "\nBuffer capacity after 5 steps: {} (allocated once, reused throughout)",
        buffer.events.capacity()
    );

    // The convenience APIs are unchanged and interchangeable with the sink path.
    let inputs = vec![0.5_f32; channels];
    let returned = encoder.encode_step(&inputs);
    let mut spikes: Vec<SpikeEvent> = Vec::new();
    encoder.encode_step_into(&inputs, &mut spikes);
    println!(
        "encode_step and encode_step_into agree: {}",
        returned.spikes == spikes
    );
}
