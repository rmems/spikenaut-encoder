use crate::prelude::*;
use rand::Rng;

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
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        for (i, &value) in input.iter().enumerate() {
            let normalized = self.normalize(value);
            let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
            let probability = (rate / 10.0).clamp(0.0, 1.0);

            if rand::thread_rng().gen_range(0.0f32..1.0) < probability {
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
        let output = encoder.encode(&input);
        
        let num_spikes = output.spikes.len();
        assert!(num_spikes <= 3);
    }
}
