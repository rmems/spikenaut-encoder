//! Rate Encoding Example
//!
//! Demonstrates GPU-accelerated rate encoding at full Blackwell warp scale.
//! 1024 float stimulus values are uploaded to the RTX 5080 in a single
//! `GpuBuffer::from_slice` call, and the `poisson_encode` CUDA kernel fires
//! across a fully-saturated warp grid in one dispatch.
//!
//! ```
//! cargo run --example rate_encoding
//! ```

use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

fn main() {
    // Initialize GPU context (owned by the parent application).
    let gpu = GpuAccelerator::new();

    // Use the Blackwell baseline config to define channel width.
    let config = EncoderConfig::default();
    println!("=== Rate Encoding (Warp-Optimized 1024-Channel) ===");
    println!(
        "Architecture: {} input channels → {} output channels",
        config.input_channels, config.output_channels
    );

    // Create a rate encoder:
    //   base_rate  = 5.0   (minimum firing rate)
    //   max_rate   = 100.0 (maximum firing rate)
    //   range      = (0.0, 1.0) (normalized input value range)
    let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));

    // Generate a 1024-element stimulus: linearly spaced values from 0.0 to 1.0.
    // This saturates a full CUDA warp grid on the RTX 5080 in a single dispatch.
    let inputs: Vec<f32> = (0..config.input_channels)
        .map(|i| i as f32 / (config.input_channels - 1) as f32)
        .collect();

    println!("Input channels: {}\n", inputs.len());

    // Encode over multiple time steps to observe stochastic behavior.
    for step in 0..5 {
        let output = encoder.encode(&inputs, &gpu);
        println!(
            "Step {}: {}/{} channels fired",
            step,
            output.spikes.len(),
            config.input_channels
        );
    }
}
