//! Predictive (Anomaly) Encoding Example
//!
//! Demonstrates predictive encoding where spikes are generated when input
//! values deviate significantly from the encoder's internal running prediction.
//! This is useful for anomaly detection in streaming sensor data.
//!
//! ```
//! cargo run --example predictive_encoding
//! ```

use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

fn main() {
    let gpu = GpuAccelerator::new();

    // Create a predictive encoder:
    //   history_depth         = 10  (rolling window for prediction)
    //   deviation_thresholds  = [(3.0, 1), (8.0, 2)]  (threshold, spike value)
    //   num_channels          = 1
    let mut encoder = PredictiveEncoder::new(10, vec![(3.0, 1), (8.0, 2)], 1);

    // Simulate a sensor stream: stable baseline, then anomalies.
    let stream: Vec<f32> = vec![
        5.0, 5.1, 4.9, 5.0, 5.2,   // Stable baseline
        5.1, 4.8, 5.0, 5.0, 5.1,   // Continued stable
        15.0,                        // Anomaly: sudden spike
        5.0, 5.0, 5.1, 4.9,        // Return to normal
        -5.0,                        // Anomaly: sudden drop
    ];

    println!("=== Predictive (Anomaly) Encoding ===");
    println!("Deviation thresholds: 3.0 (mild), 8.0 (severe)\n");

    for (step, &value) in stream.iter().enumerate() {
        let output = encoder.encode(&[value], &gpu);
        let label = if output.spikes.is_empty() {
            "normal".to_string()
        } else {
            format!("ANOMALY ({} spike(s))", output.spikes.len())
        };
        println!("Step {:2}: value = {:6.1}  -> {}", step, value, label);
    }
}
