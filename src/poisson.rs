//! Poisson spike-train encoder.
//!
//! Converts a normalized intensity value (0.0–1.0) into a binary spike train
//! of length `num_steps` using stochastic Poisson firing.
//!
//! # Physics analogy
//! Acts like a Geiger counter for your data:
//! high intensity → high click rate (spikes).

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encode a normalized value (0.0–1.0) into a temporal spike train.
    ///
    /// Each timestep independently fires with probability equal to `input`.
    /// Returns a `Vec<u8>` of 0s and 1s of length `num_steps`.
    pub fn encode(&self, input: f32) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let probability = input.clamp(0.0, 1.0);
        (0..self.num_steps)
            .map(|_| if rng.gen::<f32>() < probability { 1 } else { 0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_matches_num_steps() {
        let enc = PoissonEncoder::new(50);
        let spikes = enc.encode(0.5);
        assert_eq!(spikes.len(), 50);
    }

    #[test]
    fn zero_input_produces_no_spikes() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(0.0);
        assert!(spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn full_input_produces_all_spikes() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(1.0);
        assert!(spikes.iter().all(|&s| s == 1));
    }

    #[test]
    fn values_are_binary() {
        let enc = PoissonEncoder::new(200);
        let spikes = enc.encode(0.4);
        assert!(spikes.iter().all(|&s| s == 0 || s == 1));
    }
}
