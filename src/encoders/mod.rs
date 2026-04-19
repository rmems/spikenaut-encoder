pub mod rate;
pub mod temporal;
pub mod predictive;
pub mod population;
pub mod neuromod;
pub mod delta;

pub use rate::RateEncoder;
pub use temporal::TemporalEncoder;
pub use predictive::PredictiveEncoder;
pub use population::PopulationEncoder;
pub use delta::{DeltaEncoder, encode_deltas_to_spikes};
