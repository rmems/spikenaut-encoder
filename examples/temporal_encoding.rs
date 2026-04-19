//! Temporal Encoding Example
//!
//! Demonstrates temporal change-detection encoding. Spikes are generated when
//! the difference between recent and older input averages exceeds configured
//! thresholds — useful for detecting sudden shifts in time-series data.
//!
//! ```
//! cargo run --example temporal_encoding
//! ```

use axon_encoder::prelude::*;

fn main() {
    // Create a temporal encoder:
    //   history_depth      = 10  (number of past values to track)
    //   change_thresholds  = [(2.0, 1), (5.0, 2)]  (threshold, spike value)
    //   num_channels       = 2
    let mut encoder = TemporalEncoder::new(10, vec![(2.0, 1), (5.0, 2)], 2);

    // Simulate a stable period followed by sudden change on channel 0.
    let stable_readings = [
        [1.0, 5.0],
        [1.1, 5.1],
        [1.0, 4.9],
        [1.2, 5.0],
        [1.0, 5.2],
    ];
    let spike_readings = [
        [8.0, 5.0],  // Sudden jump on channel 0
        [9.0, 5.1],
        [8.5, 12.0], // Sudden jump on channel 1
    ];

    println!("=== Temporal Encoding ===\n");

    println!("--- Stable phase ---");
    for (i, input) in stable_readings.iter().enumerate() {
        let output = encoder.encode(input);
        println!("Step {}: {} spike(s)", i, output.spikes.len());
    }

    println!("\n--- Change phase ---");
    for (i, input) in spike_readings.iter().enumerate() {
        let output = encoder.encode(input);
        let details: Vec<String> = output
            .spikes
            .iter()
            .map(|s| format!("ch{}", s.channel))
            .collect();
        println!(
            "Step {}: {} spike(s) {:?}  (input: {:?})",
            i + stable_readings.len(),
            output.spikes.len(),
            details,
            input
        );
    }
}
