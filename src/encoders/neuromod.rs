use crate::prelude::*;
use crate::modulators::NeuroModulators;
use rand::Rng;
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

pub struct NeuromodSensoryEncoder {
    neuromodulators: NeuroModulators,
    channel_gains: Vec<f32>,
    channel_biases: Vec<f32>,
    adaptation_rates: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
}

impl NeuromodSensoryEncoder {
    pub fn new(input_channels: usize, output_channels: usize) -> Self {
        assert_eq!(output_channels % input_channels, 0, "Output channels must be a multiple of input channels.");
        Self {
            neuromodulators: NeuroModulators::default(),
            channel_gains: vec![1.0; output_channels],
            channel_biases: vec![0.0; output_channels],
            adaptation_rates: vec![0.01; output_channels],
            input_channels,
            output_channels,
        }
    }

    pub fn update_neuromodulators(&mut self, modulators: NeuroModulators) {
        self.neuromodulators = modulators;
    }

    fn calculate_rates(&self, value: f32, channel_idx: usize) -> (f32, f32) {
        let dopamine_boost = self.neuromodulators.dopamine;
        let gain = self.channel_gains[channel_idx] * (1.0 + dopamine_boost);
        let bias = self.channel_biases[channel_idx] + self.neuromodulators.cortisol * 0.2;
        
        let scaled_value = (value * gain + bias).clamp(-3.0, 3.0);
        let noise = 0.01;

        let negative_rate = (-scaled_value / 3.0).clamp(0.0, 1.0) + noise;
        let positive_rate = (scaled_value / 3.0).clamp(0.0, 1.0) + noise;
        
        (negative_rate, positive_rate)
    }

    fn update_adaptation(&mut self, stimuli: &[f32], active_channels: usize) {
        let ach_boost = 1.0 + self.neuromodulators.acetylcholine;
        let tempo_boost = 1.0 + self.neuromodulators.tempo;

        for (i, &activity) in stimuli.iter().enumerate().take(active_channels) {
            let rate = self.adaptation_rates[i] * ach_boost * tempo_boost;
            let target_rate = 0.1;
            let error = activity - target_rate;

            self.channel_gains[i] -= rate * error * 0.1;
            self.channel_gains[i] = self.channel_gains[i].clamp(0.1, 3.0);

            if activity < 0.05 {
                self.channel_biases[i] += rate * 0.01;
            } else if activity > 0.5 {
                self.channel_biases[i] -= rate * 0.01;
            }
            self.channel_biases[i] = self.channel_biases[i].clamp(-0.5, 0.5);
        }
    }
}

impl Encoder for NeuromodSensoryEncoder {
    fn encode(&mut self, input: &[f32], gpu: &GpuAccelerator) -> EncodedOutput {
        self.neuromodulators.decay(); // 1. Decay modulators
        let mut output = EncodedOutput::new();
        let channels_per_input = self.output_channels / self.input_channels;
        let tempo_scale = 1.0 + self.neuromodulators.tempo;

        let num_inputs = input.len().min(self.input_channels);
        let active_channels = num_inputs * channels_per_input;
        
        let mut stimuli = vec![0.0f32; self.output_channels];
        let mut host_probs = Vec::with_capacity(active_channels * 2);

        for (i, &value) in input.iter().take(num_inputs).enumerate() {
            for j in 0..channels_per_input {
                let output_channel = i * channels_per_input + j;
                let (negative_rate, positive_rate) = self.calculate_rates(value, output_channel);
                
                let negative_prob = negative_rate * tempo_scale;
                let positive_prob = positive_rate * tempo_scale;
                
                host_probs.push(negative_prob);
                host_probs.push(positive_prob);
                
                stimuli[output_channel] = negative_prob + positive_prob; // For adaptation
            }
        }

        self.update_adaptation(&stimuli, active_channels);
        output.embeddings = Some(stimuli);

        if host_probs.is_empty() {
            return output;
        }

        let n = host_probs.len();

        if gpu.is_ready() {
            if let (Ok(d_stimuli), Ok(mut d_spikes)) = (
                GpuBuffer::from_slice(&host_probs),
                GpuBuffer::<u32>::alloc(n),
            ) {
                let seed = rand::random::<u32>();
                if gpu.poisson_encode(&d_stimuli, &mut d_spikes, seed).is_ok() {
                    if let Ok(host_spikes) = d_spikes.to_vec() {
                        for k in 0..active_channels {
                            let output_channel = k;
                            if host_spikes[k * 2] == 1 {
                                output.spikes.push(SpikeEvent {
                                    channel: output_channel as u16,
                                    timestamp: 0,
                                    polarity: false,
                                });
                            }
                            if host_spikes[k * 2 + 1] == 1 {
                                output.spikes.push(SpikeEvent {
                                    channel: output_channel as u16,
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

        // CPU Fallback
        for k in 0..active_channels {
            let output_channel = k;
            if rand::thread_rng().gen_range(0.0f32..1.0) < host_probs[k * 2] {
                output.spikes.push(SpikeEvent {
                    channel: output_channel as u16,
                    timestamp: 0,
                    polarity: false,
                });
            }
            if rand::thread_rng().gen_range(0.0f32..1.0) < host_probs[k * 2 + 1] {
                output.spikes.push(SpikeEvent {
                    channel: output_channel as u16,
                    timestamp: 0,
                    polarity: true,
                });
            }
        }

        output
    }

    fn reset(&mut self) {
        self.neuromodulators = NeuroModulators::default();
        self.channel_gains = vec![1.0; self.output_channels];
        self.channel_biases = vec![0.0; self.output_channels];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromod_sensory_encoder() {
        let mut encoder = NeuromodSensoryEncoder::new(2, 4);
        let input = [1.0, -1.0];
        let gpu = GpuAccelerator::new();
        
        // Increase dopamine to ensure we get spikes (reproducibility)
        encoder.update_neuromodulators(NeuroModulators {
            dopamine: 10.0,
            ..NeuroModulators::default()
        });

        let output = encoder.encode(&input, &gpu);
        assert!(output.spikes.len() > 0);
        assert!(output.embeddings.is_some());
        assert_eq!(output.embeddings.unwrap().len(), 4);
    }

    #[test]
    fn test_neuromod_decay() {
        let mut encoder = NeuromodSensoryEncoder::new(1, 1);
        let gpu = GpuAccelerator::new();
        encoder.update_neuromodulators(NeuroModulators {
            dopamine: 1.0,
            ..NeuroModulators::default()
        });
        
        encoder.encode(&[0.0], &gpu);
        // After one encode, dopamine should have decayed
        assert!(encoder.neuromodulators.dopamine < 1.0);
    }
}
