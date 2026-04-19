use crate::types::{EncodedOutput, SpikeEvent};

#[derive(Clone, Debug)]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32, 
}

#[derive(Clone)]
pub struct EncoderState {
    pub membrane_potentials: Vec<f32>,
}

impl EncoderState {
    pub fn new_zeros(len: usize) -> Self {
        Self {
            membrane_potentials: vec![0.0; len],
        }
    }
}

pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Vec<f32>, 
}

impl EmbeddingRateEncoder {
    pub fn new(embeddings: &[f32], config: EmbeddingEncoderConfig) -> Self {
        let min_val = embeddings.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let epsilon = 1e-5f32;
        let safe_range = range + epsilon;

        let normalized: Vec<f32> = embeddings.iter()
            .map(|&x| (x - min_val) / safe_range)
            .collect();

        Self {
            config,
            normalized_embeddings: normalized,
        }
    }

    pub fn forward(&self, prev_state: &EncoderState) -> (EncodedOutput, EncoderState) {
        let len = self.normalized_embeddings.len();
        let mut new_potentials = prev_state.membrane_potentials.clone();
        let mut output = EncodedOutput::new();

        for i in 0..len {
            new_potentials[i] += self.normalized_embeddings[i];

            if new_potentials[i] >= self.config.v_th {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
                new_potentials[i] -= self.config.v_th; // Soft reset
            }
        }

        (output, EncoderState { membrane_potentials: new_potentials })
    }
}
