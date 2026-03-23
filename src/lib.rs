//! # spikenaut-encoder
//!
//! SNN sensory encoding pipelines for cyber-physical systems.
//!
//! ## Modules
//! - [`poisson`] — Stochastic Poisson spike-train generation
//! - [`sensory_encoder`] — 16-channel rate/temporal/predictive encoder for system telemetry
//! - [`neuromod_encoder`] — Neuromodulator-driven Poisson encoder for market data
//! - [`modulators`] — 7-system chemical neuromodulator state machine
//! - [`types`] — Lightweight telemetry data types

pub mod types;
pub mod modulators;
pub mod poisson;
pub mod sensory_encoder;
pub mod neuromod_encoder;

pub use types::{GpuTelemetry, SystemTelemetry, PoolEvent};
pub use modulators::NeuroModulators;
pub use poisson::PoissonEncoder;
pub use sensory_encoder::{SensoryEncoder, SensoryEncoderStats};
pub use neuromod_encoder::{NeuromodSensoryEncoder, ChannelStats};
