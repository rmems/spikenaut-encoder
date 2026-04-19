use crate::prelude::*;
use rand::Rng;

/// Encodes a single analog value across a population of neurons.
/// Each neuron is tuned to a preferred value within the input range.
pub struct PopulationEncoder {
    num_neurons: usize,
    input_range: (f32, f32),
    tuning_width: f32, // Controls how broadly a neuron responds to stimuli
}

impl PopulationEncoder {
    pub fn new(num_neurons: usize, input_range: (f32, f32), tuning_width: f32) -> Self {
        Self { num_neurons, input_range, tuning_width }
    }

    /// Calculates the firing rate for a neuron based on a Gaussian tuning curve.
    fn get_rate(&self, input: f32, neuron_index: usize) -> f32 {
        let range_span = self.input_range.1 - self.input_range.0;
        let preferred_value = self.input_range.0 + (neuron_index as f32 / self.num_neurons as f32) * range_span;
        
        let distance = (input - preferred_value).abs();
        // Gaussian-like response curve
        (- (distance * distance) / (2.0 * self.tuning_width * self.tuning_width)).exp()
    }
}

impl Encoder for PopulationEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        let mut rng = rand::thread_rng();

        // This encoder expects a single value in the input slice
        if let Some(&value) = input.first() {
            for i in 0..self.num_neurons {
                let rate = self.get_rate(value, i);
                if rng.gen_range(0.0..1.0) < rate {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0, // Simplified
                        polarity: true,
                    });
                }
            }
        }
        output
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_encoder() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        // Encode a value in the middle of the range.
        let input = [50.0];
        let output = encoder.encode(&input);

        // The neuron whose preferred value is closest to 50.0 should have the highest chance of firing.
        // We can't guarantee a spike due to the probabilistic nature, but we can check the rates.
        let rates: Vec<f32> = (0..10).map(|i| encoder.get_rate(50.0, i)).collect();
        let max_rate_index = rates.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

        // For a 10-neuron setup over a 0-100 range, the 5th neuron (index 4 or 5) should be near the max.
        assert!(max_rate_index == 4 || max_rate_index == 5, "Peak activity should be near the middle neuron for an input of 50.");
        assert!(output.spikes.len() <= 10);
    }
}
