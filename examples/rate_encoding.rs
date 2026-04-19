//! Rate Encoding Example

use axon_encoder::prelude::*;

fn main() {
    let config = EncoderConfig::default();
    println!("=== Rate Encoding Example ===");
    println!(
        "Architecture: {} input channels -> {} output channels",
        config.input_channels, config.output_channels
    );

    let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));

    let inputs: Vec<f32> = (0..config.input_channels)
        .map(|i| i as f32 / (config.input_channels - 1) as f32)
        .collect();

    println!("Input channels: {}\n", inputs.len());

    for step in 0..5 {
        let output = encoder.encode(&inputs);
        println!(
            "Step {}: {}/{} channels fired",
            step,
            output.spikes.len(),
            config.input_channels
        );
    }
}
