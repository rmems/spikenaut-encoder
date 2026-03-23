//! Neuromod Sensory Encoder - Spikenaut Poisson Encoding
//!
//! Replaces basic telemetry mapping with neuromodulator-driven Poisson sensory encoding.
//! Integrates all 7 neuromodulators for adaptive firing rate modulation.

use crate::modulators::NeuroModulators;
use crate::types::GpuTelemetry;

/// Neuromod Sensory Encoder for Spikenaut
///
/// Provides 16-channel Poisson encoding driven by neuromodulator state.
/// Replaces basic telemetry mapping with adaptive sensory processing.
pub struct NeuromodSensoryEncoder {
    neuromodulators: NeuroModulators,
    channel_gains: [f32; 16],
    channel_biases: [f32; 16],
    adaptation_rates: [f32; 16],
}

impl NeuromodSensoryEncoder {
    pub fn new() -> Self {
        Self {
            neuromodulators: NeuroModulators::default(),
            channel_gains: [1.0; 16],
            channel_biases: [0.0; 16],
            adaptation_rates: [0.01; 16],
        }
    }

    /// Update neuromodulator state from telemetry
    pub fn update_neuromodulators(&mut self, telemetry: &GpuTelemetry) {
        self.neuromodulators = NeuroModulators::from_telemetry(telemetry);
    }

    /// Set market volatility (injected by market_pilot)
    pub fn set_market_volatility(&mut self, volatility: f32) {
        self.neuromodulators.market_volatility = volatility.clamp(0.0, 1.0);
    }

    /// Set mining dopamine (injected by mining supervisor)
    pub fn set_mining_dopamine(&mut self, mining_dopamine: f32) {
        self.neuromodulators.mining_dopamine = mining_dopamine.clamp(-0.8, 0.8);
    }

    /// Set FPGA stress (injected by hardware monitor)
    pub fn set_fpga_stress(&mut self, fpga_stress: f32) {
        self.neuromodulators.fpga_stress = fpga_stress.clamp(0.0, 1.0);
    }

    /// Encode 8-channel market data into 16-channel Poisson stimuli.
    ///
    /// # Arguments
    /// * `market_inputs` - 8-channel normalized market data (Z-scores)
    ///
    /// # Returns
    /// * `[f32; 16]` - 16-channel Poisson firing rates (0.0–1.0)
    pub fn encode_poisson_stimuli(&mut self, market_inputs: &[f32; 8]) -> [f32; 16] {
        let mut stimuli = [0.0f32; 16];

        let learning_rate = self.compute_learning_rate_modulation();
        let stress_inhibition = self.compute_stress_inhibition();
        let focus_gain = self.compute_focus_gain();
        let tempo_modulation = self.compute_tempo_modulation();

        for i in 0..8 {
            let market_value = market_inputs[i];
            let modulated_value = market_value
                * learning_rate
                * focus_gain
                * tempo_modulation
                - stress_inhibition;

            let (bear_rate, bull_rate) = self.poisson_encode_channel(modulated_value, i);
            stimuli[i * 2]     = bear_rate;
            stimuli[i * 2 + 1] = bull_rate;
        }

        self.update_adaptation(&stimuli);
        stimuli
    }

    /// Compute learning rate modulation from dopamine systems
    pub fn compute_learning_rate_modulation(&self) -> f32 {
        let total_dopamine = (self.neuromodulators.dopamine
            + self.neuromodulators.mining_dopamine.max(0.0))
            .clamp(0.0, 1.0);
        0.1 + total_dopamine * 1.9
    }

    /// Compute stress inhibition from cortisol and market volatility
    pub fn compute_stress_inhibition(&self) -> f32 {
        let total_stress = (self.neuromodulators.cortisol
            + self.neuromodulators.market_volatility)
            .clamp(0.0, 1.0);
        total_stress * 0.8
    }

    /// Compute focus gain from acetylcholine
    pub fn compute_focus_gain(&self) -> f32 {
        0.5 + self.neuromodulators.acetylcholine * 1.0
    }

    /// Compute tempo modulation from clock speed and FPGA stress
    pub fn compute_tempo_modulation(&self) -> f32 {
        let base_tempo = self.neuromodulators.tempo.clamp(0.7, 1.3);
        let stress_penalty = 1.0 - self.neuromodulators.fpga_stress * 0.3;
        base_tempo * stress_penalty
    }

    fn poisson_encode_channel(&self, value: f32, channel_idx: usize) -> (f32, f32) {
        let gain = self.channel_gains[channel_idx];
        let bias = self.channel_biases[channel_idx];
        let scaled_value = (value * gain + bias).clamp(-3.0, 3.0);

        if scaled_value >= 0.0 {
            let bull_rate = (scaled_value / 3.0).clamp(0.0, 1.0);
            (0.01, bull_rate)
        } else {
            let bear_rate = (-scaled_value / 3.0).clamp(0.0, 1.0);
            (bear_rate, 0.01)
        }
    }

    fn update_adaptation(&mut self, stimuli: &[f32; 16]) {
        for i in 0..16 {
            let rate = self.adaptation_rates[i];
            let activity = stimuli[i];
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

    /// Get current neuromodulator state
    pub fn get_neuromodulators(&self) -> &NeuroModulators {
        &self.neuromodulators
    }

    /// Get channel statistics for debugging
    pub fn get_channel_stats(&self) -> Vec<ChannelStats> {
        (0..16)
            .map(|i| ChannelStats {
                channel: i,
                gain: self.channel_gains[i],
                bias: self.channel_biases[i],
                adaptation_rate: self.adaptation_rates[i],
            })
            .collect()
    }

    /// Reset encoder to initial state
    pub fn reset(&mut self) {
        self.neuromodulators = NeuroModulators::default();
        self.channel_gains = [1.0; 16];
        self.channel_biases = [0.0; 16];
        self.adaptation_rates = [0.01; 16];
    }
}

impl Default for NeuromodSensoryEncoder {
    fn default() -> Self { Self::new() }
}

/// Channel statistics for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub channel: usize,
    pub gain: f32,
    pub bias: f32,
    pub adaptation_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromod_encoder_basic() {
        let mut encoder = NeuromodSensoryEncoder::new();
        // Use strong inputs so they overcome background activity (0.01)
        // even with default (low-dopamine) neuromod state.
        let market_inputs = [3.0f32, -3.0, 0.0, 3.0, -3.0, 3.0, -3.0, 3.0];
        let stimuli = encoder.encode_poisson_stimuli(&market_inputs);

        assert_eq!(stimuli.len(), 16);
        for rate in &stimuli {
            assert!(*rate >= 0.0 && *rate <= 1.0, "rate {} out of range", rate);
        }
        // Bear/bull pairing for strong signals
        for i in 0..8 {
            if market_inputs[i] > 0.0 {
                assert!(stimuli[i * 2 + 1] > stimuli[i * 2],
                    "ch{}: bull ({}) should > bear ({})", i, stimuli[i * 2 + 1], stimuli[i * 2]);
            } else if market_inputs[i] < 0.0 {
                assert!(stimuli[i * 2] > stimuli[i * 2 + 1],
                    "ch{}: bear ({}) should > bull ({})", i, stimuli[i * 2], stimuli[i * 2 + 1]);
            }
        }
    }

    #[test]
    fn test_neuromodulator_modulation() {
        let mut encoder = NeuromodSensoryEncoder::new();

        encoder.neuromodulators.dopamine = 0.9;
        encoder.neuromodulators.mining_dopamine = 0.5;
        assert!(encoder.compute_learning_rate_modulation() > 1.0);

        encoder.neuromodulators.cortisol = 0.8;
        encoder.neuromodulators.market_volatility = 0.6;
        assert!(encoder.compute_stress_inhibition() > 0.5);

        encoder.neuromodulators.acetylcholine = 0.9;
        assert!(encoder.compute_focus_gain() > 1.0);
    }
}
