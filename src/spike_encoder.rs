use crate::types::{EncodedOutput, SpikeEvent};

pub struct DerivativeEncoder {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

impl DerivativeEncoder {
    /// Creates a new derivative encoder with specific thresholds for each channel
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self {
            last_values: vec![0.0; thresholds.len()],
            thresholds,
        }
    }

    pub fn encode_step(&mut self, current_values: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        
        for (i, &current_val) in current_values.iter().enumerate() {
            if i >= self.thresholds.len() { break; }
            
            let delta = current_val - self.last_values[i];
            
            // Excitatory spike on positive jump exceeding threshold
            if delta > self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
            }
            // Inhibitory/Negative spike on sudden drop
            else if delta < -self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: false,
                });
            }
            
            self.last_values[i] = current_val;
        }
        output
    }
}
