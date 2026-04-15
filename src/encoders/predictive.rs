use crate::prelude::*;
use std::collections::VecDeque;

pub struct PredictiveEncoder {
    history: Vec<VecDeque<f32>>,
    thresholds: Vec<f32>,
    history_depth: usize,
    deviation_thresholds: Vec<(f32, u16)>,
}

impl PredictiveEncoder {
    pub fn new(history_depth: usize, deviation_thresholds: Vec<(f32, u16)>, num_channels: usize) -> Self {
        Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            thresholds: vec![0.0; num_channels],
            history_depth,
            deviation_thresholds,
        }
    }
}

impl Encoder for PredictiveEncoder {
    fn encode(&mut self, input: &[f32], _gpu: &myelin_accelerator::GpuAccelerator) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            let channel_history = &mut self.history[i];
            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            if channel_history.len() < 5 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(5).sum::<f32>() / 5.0;
            self.thresholds[i] = 0.9 * self.thresholds[i] + 0.1 * recent_avg;

            let deviation = (value - self.thresholds[i]).abs();

            for &(threshold, _spike_val) in self.deviation_thresholds.iter().rev() {
                if deviation > threshold {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0, // Simplified
                        polarity: true, // Indicates a deviation spike
                    });
                    break;
                }
            }
        }
        output
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
        for threshold in self.thresholds.iter_mut() {
            *threshold = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_encoder() {
        let mut encoder = PredictiveEncoder::new(5, vec![(2.0, 1)], 1);
        let gpu = myelin_accelerator::GpuAccelerator::new();
        encoder.encode(&[1.0], &gpu);
        encoder.encode(&[1.0], &gpu);
        encoder.encode(&[1.0], &gpu);
        encoder.encode(&[1.0], &gpu);
        encoder.encode(&[1.0], &gpu);
        let output = encoder.encode(&[10.0], &gpu);
        assert!(!output.spikes.is_empty());
    }
}
