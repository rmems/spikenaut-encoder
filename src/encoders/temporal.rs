use crate::prelude::*;
use std::collections::VecDeque;

pub struct TemporalEncoder {
    history: Vec<VecDeque<f32>>,
    history_depth: usize,
    change_thresholds: Vec<(f32, u16)>,
}

impl TemporalEncoder {
    pub fn new(history_depth: usize, change_thresholds: Vec<(f32, u16)>, num_channels: usize) -> Self {
        Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            history_depth,
            change_thresholds,
        }
    }
}

impl Encoder for TemporalEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            let channel_history = &mut self.history[i];
            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            if channel_history.len() < 3 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(3).sum::<f32>() / 3.0;
            let older_avg = channel_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
            let change = (recent_avg - older_avg).abs();

            for &(threshold, _spike_val) in self.change_thresholds.iter().rev() {
                if change > threshold {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0, // Simplified
                        polarity: true, // Or use spike_val to determine polarity/strength
                    });
                    break; // Only fire one spike per channel per step
                }
            }
        }
        output
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_encoder() {
        let mut encoder = TemporalEncoder::new(6, vec![(2.0, 1), (5.0, 2)], 1);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[8.0]);
        let output = encoder.encode(&[8.0]);
        assert!(!output.spikes.is_empty());
    }
}
