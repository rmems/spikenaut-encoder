//! Embedding Rate Encoder Example
//!
//! Demonstrates the `EmbeddingRateEncoder` which normalizes raw embeddings
//! using Linear Min-Max Scaling and converts them into spike trains via a
//! leaky integrate-and-fire (LIF) mechanism. This is useful for converting
//! dense LLM embedding vectors into event-based representations for SNN
//! processing.
//!
//! ```
//! cargo run --example embedding_encoder
//! ```

use spikenaut_encoder::prelude::*;

fn main() {
    // Simulated 8-dimensional embedding vector (e.g., from an LLM).
    let raw_embeddings: Vec<f32> = vec![0.2, -1.5, 3.0, 0.8, -0.1, 2.5, 1.0, -0.5];

    // Configure the encoder with a voltage threshold.
    let config = EmbeddingEncoderConfig { v_th: 0.6 };
    let encoder = EmbeddingRateEncoder::new(&raw_embeddings, config);

    // Initialize membrane potentials to zero.
    let mut state = EncoderState::new_zeros(raw_embeddings.len());

    println!("=== Embedding Rate Encoder ===");
    println!("Embedding dimension: {}", raw_embeddings.len());
    println!("Voltage threshold (v_th): 0.6\n");

    // Run the encoder for several forward steps.
    for step in 0..10 {
        let (spikes, new_state) = encoder.forward_t(&state);
        state = new_state;

        let spike_indices: Vec<usize> = spikes
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i)
            .collect();
        let membrane: Vec<String> = state.u_enc.iter().map(|v| format!("{:.2}", v)).collect();

        println!(
            "Step {:2}: spikes at {:?}, membrane = [{}]",
            step,
            spike_indices,
            membrane.join(", ")
        );
    }
}
