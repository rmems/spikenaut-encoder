use crate::prelude::*;

/// A simple delta-based encoder.
/// Fires a spike when the absolute difference between the current input and the last encoded value exceeds a threshold.
pub struct DeltaEncoder {
    last_values: Vec<f32>,
    threshold: f32,
}

impl DeltaEncoder {
    pub fn new(threshold: f32, num_channels: usize) -> Self {
        Self {
            last_values: vec![0.0; num_channels],
            threshold,
        }
    }
}

impl Encoder for DeltaEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            if i >= self.last_values.len() {
                break;
            }
            let delta = (value - self.last_values[i]).abs();
            if delta > self.threshold {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: value > self.last_values[i],
                });
                self.last_values[i] = value;
            }
        }
        output
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

/// Simplified: delta-based spike generation (per feature)
/// 
/// This is a utility function that takes a slice of deltas and returns a boolean spike train.
/// It can be used to feed the resulting binary/event sequences into Spikenaut LIF/RSNN layers.
pub fn encode_deltas_to_spikes(deltas: &[f32], threshold: f32) -> Vec<bool> {
    deltas.iter().map(|&d| d.abs() > threshold).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new(2.0, 1);
        let output = encoder.encode(&[1.0]); // 1.0 - 0.0 = 1.0 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[3.5]); // 3.5 - 0.0 = 3.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert_eq!(output.spikes[0].polarity, true);

        let output = encoder.encode(&[4.0]); // 4.0 - 3.5 = 0.5 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[1.0]); // 1.0 - 3.5 = -2.5.abs() = 2.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert_eq!(output.spikes[0].polarity, false);
    }

    #[test]
    fn test_encode_deltas_to_spikes() {
        let deltas = [0.1, 0.5, -0.8, 1.2];
        let threshold = 0.7;
        let spikes = encode_deltas_to_spikes(&deltas, threshold);
        assert_eq!(spikes, vec![false, false, true, true]);
    }
}
