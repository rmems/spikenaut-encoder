//! Delta Encoding Example
//!
//! Demonstrates delta-based spike encoding. A spike fires when the absolute
//! difference between the current and last-encoded value exceeds a threshold.
//! Polarity indicates the direction of change (true = increase, false = decrease).
//!
//! ```
//! cargo run --example delta_encoding
//! ```

use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

fn main() {
    let gpu = GpuAccelerator::new();

    // Create a delta encoder:
    //   threshold    = 3.0  (minimum change magnitude to trigger a spike)
    //   num_channels = 2
    let mut encoder = DeltaEncoder::new(3.0, 2);

    // Simulated two-channel time series with occasional large jumps.
    let readings: Vec<[f32; 2]> = vec![
        [0.0, 10.0],
        [1.0, 10.5],   // Small changes — no spikes
        [5.0, 10.2],   // Channel 0 jumps +5.0 from last encoded (0.0)
        [5.5, 10.0],   // Small change from 5.0
        [1.0, 20.0],   // Channel 0 drops -4.5 from 5.0; channel 1 jumps +10.0
        [1.5, 20.5],   // Small changes
    ];

    println!("=== Delta Encoding ===");
    println!("Threshold: 3.0\n");

    for (step, input) in readings.iter().enumerate() {
        let output = encoder.encode(input, &gpu);
        if output.spikes.is_empty() {
            println!("Step {}: input {:?} -> no spikes", step, input);
        } else {
            for spike in &output.spikes {
                let direction = if spike.polarity { "↑" } else { "↓" };
                println!(
                    "Step {}: input {:?} -> SPIKE on channel {} {}",
                    step, input, spike.channel, direction
                );
            }
        }
    }

    // Also demonstrate the utility function for raw delta arrays.
    println!("\n--- encode_deltas_to_spikes utility ---");
    let deltas = [0.1, -0.5, 3.2, -4.0, 0.8];
    let threshold = 1.0;
    let spikes = encode_deltas_to_spikes(
        &deltas.iter().map(|&d| d as f64).collect::<Vec<_>>(),
        threshold as f64,
    );
    println!("Deltas:    {:?}", deltas);
    println!("Threshold: {}", threshold);
    println!("Spikes:    {:?}", spikes);
}
