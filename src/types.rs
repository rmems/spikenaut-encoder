//! Standardized types for encoder inputs and outputs.

/// A single spike event.
#[derive(Clone, Copy, Debug)]
pub struct SpikeEvent {
    pub channel: u16,
    pub timestamp: u64,   // or relative step
    pub polarity: bool,   // or strength
}

/// Optional metadata about the encoding process.
#[derive(Clone, Debug, Default)]
pub struct EncodingMetadata {
    // Add any relevant metadata fields here, e.g.:
    // pub source_sample_index: u64,
}

/// The standardized output of an encoder.
#[derive(Default)]
pub struct EncodedOutput {
    pub spikes: Vec<SpikeEvent>,
    pub embeddings: Option<Vec<f32>>,
    pub metadata: Option<EncodingMetadata>,
}

impl EncodedOutput {
    pub fn new() -> Self {
        Self::default()
    }
}

/// General-purpose configuration for encoders.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub input_channels: usize,
    pub output_channels: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 256,
            output_channels: 256,
        }
    }
}
