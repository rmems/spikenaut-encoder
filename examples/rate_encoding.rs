//! Rate Encoding Example
//!
//! Demonstrates GPU-accelerated rate encoding where continuous sensor values
//! are converted into probabilistic spike trains. Higher input values produce
//! higher firing rates.
//!
//! ```
//! cargo run --example rate_encoding
//! ```

use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

fn main() {
    // Initialize GPU context (owned by the parent application).
    let gpu = GpuAccelerator::new();

    // Create a rate encoder:
    //   base_rate  = 5.0  (minimum firing rate)
    //   max_rate   = 100.0 (maximum firing rate)
    //   range      = (0.0, 100.0) (expected input value range)
    let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 100.0));

    // Simulated 4-channel sensor readings.
    let inputs = [10.0, 40.0, 75.0, 95.0];

    println!("=== Rate Encoding ===");
    println!("Input values: {:?}\n", inputs);

    // Encode over multiple time steps to observe stochastic behavior.
    for step in 0..5 {
        let output = encoder.encode(&inputs, &gpu);
        let channels: Vec<u16> = output.spikes.iter().map(|s| s.channel).collect();
        println!(
            "Step {}: {} spike(s) on channels {:?}",
            step,
            output.spikes.len(),
            channels
        );
    }
}
