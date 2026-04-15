use corinth_canal::tensor::{Tensor, zeros};

#[derive(Clone, Debug)]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32, 
}

#[derive(Clone)]
pub struct EncoderState {
    pub u_enc: Tensor, 
}

impl EncoderState {
    pub fn new_zeros(len: usize) -> Self {
        Self {
            u_enc: zeros(len),
        }
    }
}

pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Tensor, 
}

impl EmbeddingRateEncoder {
    /// Ingests OLMo embeddings and applies Linear Min-Max Scaling
    /// to preserve spatial relationships between tokens.
    pub fn new(embeddings: &[f32], config: EmbeddingEncoderConfig) -> Self {
        // Linear Min-Max Scaling: (X - X_min) / (X_max - X_min)
        
        let min_val = embeddings.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        let range = max_val - min_val;
        
        // Add a tiny epsilon to prevent division by zero in dead neurons
        let epsilon = 1e-5f32;
        let safe_range = range + epsilon;

        let normalized: Tensor = embeddings.iter()
            .map(|&x| (x - min_val) / safe_range)
            .collect();

        Self {
            config,
            normalized_embeddings: normalized,
        }
    }

    /// The highly optimized forward step. 
    /// Note: In production, this block will be replaced by a custom CUDA pass.
    pub fn forward_t(&self, prev_state: &EncoderState) -> (Tensor, EncoderState) {
        let len = self.normalized_embeddings.len();
        let mut new_u_enc = zeros(len);
        let mut spikes = zeros(len);

        for i in 0..len {
            // Accumulate current
            let mut updated_u = prev_state.u_enc[i] + self.normalized_embeddings[i];

            // Evaluate spikes
            if updated_u >= self.config.v_th {
                spikes[i] = 1.0;
                updated_u -= self.config.v_th; // Soft reset
            } else {
                spikes[i] = 0.0;
            }

            new_u_enc[i] = updated_u;
        }

        // Return the boolean/f32 spikes and the updated state
        (spikes, EncoderState { u_enc: new_u_enc })
    }
}
