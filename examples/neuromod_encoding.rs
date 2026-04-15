//! Neuromodulatory Sensory Encoding Example
//!
//! Demonstrates the `NeuromodSensoryEncoder` which uses biological
//! neuromodulator states (Dopamine, Cortisol, Acetylcholine, Tempo) to
//! dynamically adjust gain, bias, and adaptation of spike encoding.
//!
//! ```
//! cargo run --example neuromod_encoding
//! ```

use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

fn main() {
    let gpu = GpuAccelerator::new();

    // Create a neuromodulatory encoder:
    //   input_channels  = 2
    //   output_channels = 4  (2x expansion per input channel)
    let mut encoder = NeuromodSensoryEncoder::new(2, 4);

    let input = [0.5, -0.3];

    println!("=== Neuromodulatory Sensory Encoding ===\n");

    // --- Baseline encoding (no modulator boost) ---
    println!("--- Baseline (default modulators) ---");
    for step in 0..3 {
        let output = encoder.encode(&input, &gpu);
        println!(
            "Step {}: {} spike(s), embeddings length = {}",
            step,
            output.spikes.len(),
            output.embeddings.as_ref().map_or(0, |e| e.len())
        );
    }

    // --- Boost dopamine: increases gain/sensitivity ---
    println!("\n--- High dopamine (reward signal) ---");
    encoder.reset();
    encoder.update_neuromodulators(NeuroModulators {
        dopamine: 2.0,
        cortisol: 0.0,
        acetylcholine: 0.0,
        tempo: 0.0,
    });
    for step in 0..3 {
        let output = encoder.encode(&input, &gpu);
        println!(
            "Step {}: {} spike(s)",
            step,
            output.spikes.len()
        );
    }

    // --- Boost cortisol: modulates channel biases ---
    println!("\n--- High cortisol (stress signal) ---");
    encoder.reset();
    encoder.update_neuromodulators(NeuroModulators {
        dopamine: 0.0,
        cortisol: 2.0,
        acetylcholine: 0.0,
        tempo: 0.0,
    });
    for step in 0..3 {
        let output = encoder.encode(&input, &gpu);
        println!(
            "Step {}: {} spike(s)",
            step,
            output.spikes.len()
        );
    }

    // --- Observe natural modulator decay ---
    println!("\n--- Modulator decay over time ---");
    encoder.reset();
    encoder.update_neuromodulators(NeuroModulators {
        dopamine: 5.0,
        cortisol: 3.0,
        acetylcholine: 2.0,
        tempo: 1.0,
    });
    for step in 0..10 {
        let output = encoder.encode(&input, &gpu);
        println!(
            "Step {:2}: {} spike(s)",
            step,
            output.spikes.len()
        );
    }
}
