//! Sensory Encoder - Converting Continuous Telemetry to Spike Trains
//!
//! Transforms floating-point telemetry values into biologically realistic
//! spike trains that the Spiking Neural Network can process.
//!
//! ENCODING STRATEGIES:
//! - Rate Coding: Higher values → higher spike rates
//! - Temporal Coding: Value changes → spike timing patterns
//! - Population Coding: Multiple neurons encode different ranges
//! - Predictive Coding: Deviations from expected patterns

use std::collections::VecDeque;
use crate::types::SystemTelemetry;

/// Sensory encoder that converts telemetry to spike trains
pub struct SensoryEncoder {
    // History for temporal coding and change detection
    cpu_temp_history: VecDeque<f32>,
    cpu_power_history: VecDeque<f32>,
    gpu_temp_history: VecDeque<f32>,
    gpu_power_history: VecDeque<f32>,

    // Expected ranges for normalization
    cpu_temp_range: (f32, f32),   // 40°C - 90°C typical
    cpu_power_range: (f32, f32),  // 20W - 150W typical
    gpu_temp_range: (f32, f32),   // 30°C - 85°C typical
    gpu_power_range: (f32, f32),  // 50W - 450W typical

    // Spike generation parameters
    history_depth: usize,
    base_rate: f32,   // Baseline spike rate (Hz)
    max_rate: f32,    // Maximum spike rate (Hz)

    // Adaptive thresholds for predictive coding
    cpu_temp_threshold: f32,
    cpu_power_threshold: f32,
    gpu_temp_threshold: f32,
    gpu_power_threshold: f32,
}

impl SensoryEncoder {
    /// Create new sensory encoder with biologically realistic parameters
    pub fn new() -> Self {
        Self {
            cpu_temp_history: VecDeque::with_capacity(10),
            cpu_power_history: VecDeque::with_capacity(10),
            gpu_temp_history: VecDeque::with_capacity(10),
            gpu_power_history: VecDeque::with_capacity(10),

            cpu_temp_range: (40.0, 90.0),
            cpu_power_range: (20.0, 150.0),
            gpu_temp_range: (30.0, 85.0),
            gpu_power_range: (50.0, 450.0),

            history_depth: 10,
            base_rate: 5.0,   // 5 Hz baseline firing
            max_rate: 100.0,  // 100 Hz maximum firing

            cpu_temp_threshold: 0.0,
            cpu_power_threshold: 0.0,
            gpu_temp_threshold: 0.0,
            gpu_power_threshold: 0.0,
        }
    }

    /// Encode system telemetry into 16-channel spike train
    ///
    /// CHANNEL MAPPING:
    /// 0-3: CPU telemetry (temp, power, ccd1, ccd2)
    /// 4-7: GPU telemetry (temp, power, voltage, fan)
    /// 8-11: CPU change detection (delta from expected)
    /// 12-15: GPU change detection (delta from expected)
    pub fn encode_system_telemetry(&mut self, telemetry: &SystemTelemetry) -> [u16; 16] {
        let mut spikes = [0u16; 16];

        self.update_histories(telemetry);

        // CPU telemetry channels (0-3)
        spikes[0] = self.rate_encode_temp(telemetry.cpu_tctl_c, self.cpu_temp_range);
        spikes[1] = self.rate_encode_power(telemetry.cpu_package_power_w, self.cpu_power_range);
        spikes[2] = self.rate_encode_temp(telemetry.cpu_ccd1_c, self.cpu_temp_range);
        spikes[3] = self.rate_encode_temp(telemetry.cpu_ccd2_c, self.cpu_temp_range);

        // GPU telemetry channels (4-7)
        spikes[4] = self.rate_encode_temp(telemetry.gpu_temp_c, self.gpu_temp_range);
        spikes[5] = self.rate_encode_power(telemetry.gpu_power_w, self.gpu_power_range);
        spikes[6] = self.rate_encode_voltage(telemetry.vddcr_gfx_v);
        spikes[7] = self.rate_encode_fan_speed(telemetry.fan_speed_pct);

        // CPU change detection channels (8-11)
        spikes[8]  = self.temporal_encode_cpu_temp_change();
        spikes[9]  = self.temporal_encode_cpu_power_change();
        spikes[10] = self.predictive_encode_cpu_temp();
        spikes[11] = self.predictive_encode_cpu_power();

        // GPU change detection channels (12-15)
        spikes[12] = self.temporal_encode_gpu_temp_change();
        spikes[13] = self.temporal_encode_gpu_power_change();
        spikes[14] = self.predictive_encode_gpu_temp();
        spikes[15] = self.predictive_encode_gpu_power();

        spikes
    }

    fn rate_encode_temp(&self, temp: f32, range: (f32, f32)) -> u16 {
        let normalized = self.normalize_value(temp, range);
        let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
        self.poisson_spike(rate)
    }

    fn rate_encode_power(&self, power: f32, range: (f32, f32)) -> u16 {
        let normalized = self.normalize_value(power, range);
        let log_normalized = normalized.sqrt();
        let rate = self.base_rate + log_normalized * (self.max_rate - self.base_rate);
        self.poisson_spike(rate)
    }

    fn rate_encode_voltage(&self, voltage: f32) -> u16 {
        let normalized = ((voltage - 0.7) / 0.4).clamp(0.0, 1.0);
        let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
        self.poisson_spike(rate)
    }

    fn rate_encode_fan_speed(&self, fan_pct: f32) -> u16 {
        let normalized = (fan_pct / 100.0).clamp(0.0, 1.0);
        let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
        self.poisson_spike(rate)
    }

    fn temporal_encode_cpu_temp_change(&self) -> u16 {
        if self.cpu_temp_history.len() < 3 { return 0; }
        let recent_avg = self.cpu_temp_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self.cpu_temp_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
        let change = (recent_avg - older_avg).abs();
        if change > 5.0 { 5 } else if change > 2.0 { 3 } else if change > 0.5 { 1 } else { 0 }
    }

    fn temporal_encode_cpu_power_change(&self) -> u16 {
        if self.cpu_power_history.len() < 3 { return 0; }
        let recent_avg = self.cpu_power_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self.cpu_power_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
        let change = (recent_avg - older_avg).abs();
        if change > 20.0 { 5 } else if change > 10.0 { 3 } else if change > 2.0 { 1 } else { 0 }
    }

    fn temporal_encode_gpu_temp_change(&self) -> u16 {
        if self.gpu_temp_history.len() < 3 { return 0; }
        let recent_avg = self.gpu_temp_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self.gpu_temp_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
        let change = (recent_avg - older_avg).abs();
        if change > 5.0 { 5 } else if change > 2.0 { 3 } else if change > 0.5 { 1 } else { 0 }
    }

    fn temporal_encode_gpu_power_change(&self) -> u16 {
        if self.gpu_power_history.len() < 3 { return 0; }
        let recent_avg = self.gpu_power_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self.gpu_power_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
        let change = (recent_avg - older_avg).abs();
        if change > 50.0 { 5 } else if change > 20.0 { 3 } else if change > 5.0 { 1 } else { 0 }
    }

    fn predictive_encode_cpu_temp(&mut self) -> u16 {
        if self.cpu_temp_history.len() < 5 { return 0; }
        let recent_avg = self.cpu_temp_history.iter().rev().take(5).sum::<f32>() / 5.0;
        self.cpu_temp_threshold = 0.9 * self.cpu_temp_threshold + 0.1 * recent_avg;
        let current = self.cpu_temp_history.back().unwrap_or(&0.0);
        let deviation = (current - self.cpu_temp_threshold).abs();
        if deviation > 3.0 { 3 } else if deviation > 1.0 { 1 } else { 0 }
    }

    fn predictive_encode_cpu_power(&mut self) -> u16 {
        if self.cpu_power_history.len() < 5 { return 0; }
        let recent_avg = self.cpu_power_history.iter().rev().take(5).sum::<f32>() / 5.0;
        self.cpu_power_threshold = 0.9 * self.cpu_power_threshold + 0.1 * recent_avg;
        let current = self.cpu_power_history.back().unwrap_or(&0.0);
        let deviation = (current - self.cpu_power_threshold).abs();
        if deviation > 15.0 { 3 } else if deviation > 5.0 { 1 } else { 0 }
    }

    fn predictive_encode_gpu_temp(&mut self) -> u16 {
        if self.gpu_temp_history.len() < 5 { return 0; }
        let recent_avg = self.gpu_temp_history.iter().rev().take(5).sum::<f32>() / 5.0;
        self.gpu_temp_threshold = 0.9 * self.gpu_temp_threshold + 0.1 * recent_avg;
        let current = self.gpu_temp_history.back().unwrap_or(&0.0);
        let deviation = (current - self.gpu_temp_threshold).abs();
        if deviation > 3.0 { 3 } else if deviation > 1.0 { 1 } else { 0 }
    }

    fn predictive_encode_gpu_power(&mut self) -> u16 {
        if self.gpu_power_history.len() < 5 { return 0; }
        let recent_avg = self.gpu_power_history.iter().rev().take(5).sum::<f32>() / 5.0;
        self.gpu_power_threshold = 0.9 * self.gpu_power_threshold + 0.1 * recent_avg;
        let current = self.gpu_power_history.back().unwrap_or(&0.0);
        let deviation = (current - self.gpu_power_threshold).abs();
        if deviation > 30.0 { 3 } else if deviation > 10.0 { 1 } else { 0 }
    }

    fn update_histories(&mut self, telemetry: &SystemTelemetry) {
        self.cpu_temp_history.push_back(telemetry.cpu_tctl_c);
        self.cpu_power_history.push_back(telemetry.cpu_package_power_w);
        self.gpu_temp_history.push_back(telemetry.gpu_temp_c);
        self.gpu_power_history.push_back(telemetry.gpu_power_w);

        if self.cpu_temp_history.len() > self.history_depth { self.cpu_temp_history.pop_front(); }
        if self.cpu_power_history.len() > self.history_depth { self.cpu_power_history.pop_front(); }
        if self.gpu_temp_history.len() > self.history_depth { self.gpu_temp_history.pop_front(); }
        if self.gpu_power_history.len() > self.history_depth { self.gpu_power_history.pop_front(); }
    }

    fn normalize_value(&self, value: f32, range: (f32, f32)) -> f32 {
        ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0)
    }

    fn poisson_spike(&self, rate_hz: f32) -> u16 {
        let spike_probability = (rate_hz / 10.0).clamp(0.0, 1.0);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < spike_probability { 1 } else { 0 }
    }

    /// Get current encoding statistics for debugging
    pub fn get_stats(&self) -> SensoryEncoderStats {
        SensoryEncoderStats {
            cpu_temp_avg: self.cpu_temp_history.iter().sum::<f32>() / self.cpu_temp_history.len().max(1) as f32,
            cpu_power_avg: self.cpu_power_history.iter().sum::<f32>() / self.cpu_power_history.len().max(1) as f32,
            gpu_temp_avg: self.gpu_temp_history.iter().sum::<f32>() / self.gpu_temp_history.len().max(1) as f32,
            gpu_power_avg: self.gpu_power_history.iter().sum::<f32>() / self.gpu_power_history.len().max(1) as f32,
            cpu_temp_threshold: self.cpu_temp_threshold,
            cpu_power_threshold: self.cpu_power_threshold,
            gpu_temp_threshold: self.gpu_temp_threshold,
            gpu_power_threshold: self.gpu_power_threshold,
        }
    }
}

impl Default for SensoryEncoder {
    fn default() -> Self { Self::new() }
}

/// Statistics from the sensory encoder
#[derive(Debug, Clone)]
pub struct SensoryEncoderStats {
    pub cpu_temp_avg: f32,
    pub cpu_power_avg: f32,
    pub gpu_temp_avg: f32,
    pub gpu_power_avg: f32,
    pub cpu_temp_threshold: f32,
    pub cpu_power_threshold: f32,
    pub gpu_temp_threshold: f32,
    pub gpu_power_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory_encoder_creation() {
        let encoder = SensoryEncoder::new();
        assert_eq!(encoder.history_depth, 10);
        assert_eq!(encoder.base_rate, 5.0);
        assert_eq!(encoder.max_rate, 100.0);
    }

    #[test]
    fn test_spike_generation() {
        let mut encoder = SensoryEncoder::new();
        let telemetry = SystemTelemetry {
            cpu_tctl_c: 65.0,
            cpu_package_power_w: 45.0,
            gpu_temp_c: 70.0,
            gpu_power_w: 250.0,
            ..Default::default()
        };
        let spikes = encoder.encode_system_telemetry(&telemetry);
        assert_eq!(spikes.len(), 16);
        for &spike in &spikes {
            assert!(spike <= 5); // max burst value
        }
    }

    #[test]
    fn test_temporal_encoding() {
        let mut encoder = SensoryEncoder::new();
        for _ in 0..5 {
            encoder.encode_system_telemetry(&SystemTelemetry { cpu_tctl_c: 65.0, ..Default::default() });
        }
        // Sudden temperature jump should trigger change detection
        let spikes = encoder.encode_system_telemetry(&SystemTelemetry { cpu_tctl_c: 75.0, ..Default::default() });
        assert!(spikes[8] > 0);
    }

    #[test]
    fn test_gpu_power_threshold_tracks_gpu_power_stream() {
        let mut encoder = SensoryEncoder::new();
        for _ in 0..20 {
            encoder.encode_system_telemetry(&SystemTelemetry {
                cpu_package_power_w: 30.0,
                gpu_power_w: 420.0,
                ..Default::default()
            });
        }
        let stats = encoder.get_stats();
        assert!(
            stats.gpu_power_threshold > 250.0,
            "gpu threshold should track GPU power stream, got {}",
            stats.gpu_power_threshold
        );
        assert!(
            stats.gpu_power_threshold > stats.cpu_power_threshold + 150.0,
            "gpu threshold should be distinct from cpu threshold (gpu={}, cpu={})",
            stats.gpu_power_threshold,
            stats.cpu_power_threshold
        );
    }
}
