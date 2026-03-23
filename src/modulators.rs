use serde::{Deserialize, Serialize};
use crate::types::{PoolEvent, GpuTelemetry};

// ── Decay constants (per tick, called once per second) ──────────────────────
/// Mining dopamine decays slower than event dopamine because mining state
/// changes on the order of seconds (hashrate/thermal), not sub-second
/// like pool events (ShareAccepted, BlockFound).
const MINING_DOPAMINE_DECAY: f32 = 0.97;
const EVENT_DOPAMINE_DECAY: f32 = 0.95;
const CORTISOL_DECAY: f32 = 0.90;
const ACETYLCHOLINE_DECAY: f32 = 0.99;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,      // Reward / Learning Rate (0.0 - 1.0)
    pub cortisol: f32,      // Stress / Inhibition (0.0 - 1.0)
    pub acetylcholine: f32, // Focus / Signal-to-Noise (0.0 - 1.0)
    pub tempo: f32,         // Clock-driven timing scale (0.0 - 2.0, 1.0 = nominal)
    /// FPGA timing stress — 0.0 = no violations, 1.0 = critical (WNS ≤ -5 ns)
    #[serde(default)]
    pub fpga_stress: f32,
    /// Financial market volatility injected by market_pilot.
    ///
    /// Blended into the SNN's stress_multiplier in engine.step() alongside
    /// hardware cortisol, so high market turbulence inhibits stimulus scale
    /// the same way thermal/power stress does.
    ///
    /// Range: 0.0 (calm market) – 1.0 (extreme volatility).
    /// Default 0.0 preserves backward compatibility when not running market_pilot.
    #[serde(default)]
    pub market_volatility: f32,

    /// Mining-efficiency dopamine signal computed by `MiningRewardState`.
    ///
    /// Separate from the primary `dopamine` field to preserve the existing
    /// event-driven reward pathway (ShareAccepted, BlockFound, etc.).
    /// Blended into the STDP learning rate in `engine.step()` as a survival
    /// gate: positive values validate current neural patterns, negative
    /// values suppress plasticity during thermal/power stress.
    ///
    /// Range: \[-0.8, 0.8\].  Set externally by the live supervisor each tick.
    #[serde(default)]
    pub mining_dopamine: f32,
}

impl NeuroModulators {
    /// Decode telemetry into chemical signals
    pub fn from_telemetry(telem: &GpuTelemetry) -> Self {
        // DOPAMINE: Proportional to hashrate (Reward for doing work)
        // Target: 0.0105 MH/s = 1.0 Dopamine (calibrated to actual RTX 5080 Dynex hashrate).
        // Software fallback baseline: 0.3 (raised from 0.2 to keep SNN active in
        // software-only mode — prevents zero-activity during GPU-less fallback).
        let dopamine = (telem.hashrate_mh / 0.0105).clamp(0.3, 1.0);

        // CORTISOL: Stress from heat or power spikes
        // GPU temp comes from NVML — reliable on RTX 5080.
        // Onset at 83°C (just below thermal throttle), full stress at 93°C.
        // Guard: only activate when gpu_temp_c > 1.0 (zero means sensor absent).
        let heat_stress: f32 = if telem.gpu_temp_c > 1.0 {
            ((telem.gpu_temp_c - 83.0) / 10.0).clamp(0.0, 1.0)
        } else {
            0.0 // Sensor absent — don't inject phantom stress
        };
        // RTX 5080 TDP ~430W. Stress starts at 400W (true thermal/power runaway territory).
        let power_stress = ((telem.power_w - 400.0) / 50.0).clamp(0.0, 1.0);
        // INTEL COOLANT: High-confidence Ocean Predictoor signal reduces cortisol.
        let intel_coolant = if telem.ocean_intel > 0.01 {
            (telem.ocean_intel - 0.5).abs() * 2.0 * 0.3
        } else {
            0.0
        };
        let cortisol = (heat_stress.max(power_stress) - intel_coolant).max(0.0);

        // ACETYLCHOLINE: Stability of Vcore (Focus)
        let vddcr_dev = (telem.vddcr_gfx_v - 1.0).abs();
        let acetylcholine = (1.0 - vddcr_dev * 2.0).clamp(0.4, 1.0);

        // TEMPO: Clock-driven temporal scaling
        // Nominal RTX 5080 Core Clock: 2640 MHz
        let tempo = (telem.gpu_clock_mhz / 2640.0).clamp(0.5, 2.0);

        Self {
            dopamine,
            cortisol,
            acetylcholine,
            tempo,
            fpga_stress:       0.0,
            market_volatility: 0.0,
            mining_dopamine:   0.0,
        }
    }

    /// Update chemical levels based on instantaneous events
    pub fn apply_event(&mut self, event: &PoolEvent) {
        match event {
            PoolEvent::ShareAccepted { latency_ms } => {
                self.dopamine = (self.dopamine + 0.2).min(1.0);
                if *latency_ms > 100 {
                    self.acetylcholine = (self.acetylcholine - 0.1).max(0.0);
                } else {
                    self.acetylcholine = (self.acetylcholine + 0.05).min(1.0);
                }
            }
            PoolEvent::BlockFound { .. } => {
                self.dopamine = 1.0;
            }
            PoolEvent::PoolSwitch { .. } => {
                self.cortisol = (self.cortisol + 0.3).min(1.0);
                self.dopamine = 0.0;
            }
            _ => {}
        }
    }

    /// Natural decay (Homeostasis) - Call this every second
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * EVENT_DOPAMINE_DECAY).max(0.0);
        self.cortisol = (self.cortisol * CORTISOL_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.mining_dopamine *= MINING_DOPAMINE_DECAY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_telem(gpu_temp_c: f32, power_w: f32, hashrate_mh: f32) -> GpuTelemetry {
        GpuTelemetry {
            gpu_temp_c,
            power_w,
            hashrate_mh,
            vddcr_gfx_v: 1.0,
            gpu_clock_mhz: 2640.0,
            ..Default::default()
        }
    }

    #[test]
    fn heat_stress_activates_above_threshold() {
        let m = NeuroModulators::from_telemetry(&make_telem(90.0, 300.0, 0.01));
        assert!(m.cortisol > 0.0, "90°C should produce cortisol, got {}", m.cortisol);
    }

    #[test]
    fn heat_stress_zero_at_normal_temps() {
        let m = NeuroModulators::from_telemetry(&make_telem(72.0, 300.0, 0.01));
        assert!(m.cortisol < 0.01, "72°C / 300W should produce near-zero cortisol, got {}", m.cortisol);
    }

    #[test]
    fn heat_stress_zero_when_sensor_absent() {
        let m = NeuroModulators::from_telemetry(&make_telem(0.0, 300.0, 0.01));
        assert!(m.cortisol < 0.01, "absent sensor should not inject stress, got {}", m.cortisol);
    }

    #[test]
    fn decay_constants_ordering() {
        assert!(MINING_DOPAMINE_DECAY > EVENT_DOPAMINE_DECAY);
        assert!(ACETYLCHOLINE_DECAY > MINING_DOPAMINE_DECAY);
    }

    #[test]
    fn mining_dopamine_decay_preserves_sign() {
        let mut m = NeuroModulators::default();
        m.mining_dopamine = -0.5;
        m.decay();
        assert!(m.mining_dopamine < 0.0);
        assert!(m.mining_dopamine > -0.5);
    }
}
