//! Poisson Encoding Example
//!
//! Demonstrates the standalone `PoissonEncoder` which converts a probability
//! value (0.0–1.0) into a binary spike train over a configurable number of
//! time steps. Each step independently fires with the given probability.
//!
//! ```
//! cargo run --example poisson_encoding
//! ```

use spikenaut_encoder::poisson::PoissonEncoder;

fn main() {
    let encoder = PoissonEncoder::new(50);

    let probabilities = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0];

    println!("=== Poisson Encoding ===");
    println!("Time steps: 50\n");

    for &prob in &probabilities {
        let spikes = encoder.encode(prob);
        let spike_count: u8 = spikes.iter().sum();
        let train: String = spikes
            .iter()
            .map(|&s| if s == 1 { '█' } else { '·' })
            .collect();
        println!(
            "P={:.1}  spikes={:2}/50  train: {}",
            prob, spike_count, train
        );
    }
}
