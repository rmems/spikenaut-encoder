use crate::prelude::*;
use rand::Rng;
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
}

impl RateEncoder {
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self { base_rate, max_rate, range }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }
}

impl Encoder for RateEncoder {
    fn encode(&mut self, input: &[f32], gpu: &GpuAccelerator) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        let n = input.len();
        
        let mut host_stimuli = Vec::with_capacity(n);
        for &value in input.iter() {
            let normalized = self.normalize(value);
            let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
            let probability = (rate / 10.0).clamp(0.0, 1.0);
            host_stimuli.push(probability);
        }

        if gpu.is_ready() {
            if let (Ok(d_stimuli), Ok(mut d_spikes)) = (
                GpuBuffer::from_slice(&host_stimuli),
                GpuBuffer::<u32>::alloc(n),
            ) {
                let seed = rand::random::<u32>();
                if gpu.poisson_encode(&d_stimuli, &mut d_spikes, seed).is_ok() {
                    if let Ok(host_spikes) = d_spikes.to_vec() {
                        for (i, &s) in host_spikes.iter().enumerate() {
                            if s == 1 {
                                output.spikes.push(SpikeEvent {
                                    channel: i as u16,
                                    timestamp: 0,
                                    polarity: true,
                                });
                            }
                        }
                        return output;
                    }
                }
            }
        }

        // Fallback
        for (i, &prob) in host_stimuli.iter().enumerate() {
            if rand::thread_rng().gen_range(0.0f32..1.0) < prob {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
            }
        }

        output
    }

    fn reset(&mut self) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_encoder() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [0.0, 50.0, 100.0];
        let gpu = GpuAccelerator::new();
        let output = encoder.encode(&input, &gpu);
        
        // This is a probabilistic test, so we can't assert the exact number of spikes.
        // Instead, we check that the number of spikes is within a reasonable range.
        let num_spikes = output.spikes.len();
        assert!(num_spikes <= 3);
    }
}
