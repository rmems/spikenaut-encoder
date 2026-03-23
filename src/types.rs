//! Lightweight telemetry types for sensory encoding.
//!
//! Decoupled from hardware-collection drivers (NVML, hwmon, powercap).
//! Populate these from whatever telemetry source you have and pass them
//! to the encoders.

use serde::{Deserialize, Serialize};

/// Pool events from mining supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolEvent {
    ShareAccepted { latency_ms: u32 },
    BlockFound { block_hash: String },
    PoolSwitch { from: String, to: String },
    ShareRejected { reason: String },
}

/// GPU telemetry snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuTelemetry {
    pub vddcr_gfx_v: f32,
    pub gpu_temp_c: f32,
    pub hashrate_mh: f32,
    pub power_w: f32,
    pub gpu_clock_mhz: f32,
    pub mem_clock_mhz: f32,
    pub fan_speed_pct: f32,
    pub ocean_intel: f32,
}

/// Unified system telemetry (CPU + GPU).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemTelemetry {
    // GPU fields
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub vddcr_gfx_v: f32,
    pub fan_speed_pct: f32,
    pub gpu_clock_mhz: f32,
    pub mem_clock_mhz: f32,
    pub mem_util_pct: f32,

    // CPU fields
    pub cpu_tctl_c: f32,
    pub cpu_ccd1_c: f32,
    pub cpu_ccd2_c: f32,
    pub cpu_package_power_w: f32,

    // Motherboard fields
    pub vrm_temp_c: f32,
    pub motherboard_temp_c: f32,
    pub cpu_fan_rpm: f32,

    pub timestamp_ms: u64,
}
