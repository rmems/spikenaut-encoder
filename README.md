# Axon Encoder

**A flexible sensory encoding library for spiking neural networks.**

`axon-encoder` converts continuous time-series data into biologically plausible spike trains. It serves as a front-end sensory layer for SNNs, providing a variety of encoding strategies.

## Features

- **Core Encoders**: Rate, Temporal, and Derivative encoding strategies.
- **Extensible**: Easily extendable with custom encoders and modulators.
- **Lightweight**: Minimal dependencies and a simple, clean API.
- **Standardized Output**: `EncodedOutput` provides discrete `SpikeEvent` sequences.

## Installation

```toml
[dependencies]
axon-encoder = { path = "../axon-encoder" }
```

## Quick Start

```rust
use axon_encoder::prelude::*;

// 1. Load the default configuration
let config = EncoderConfig::default();

// 2. Initialize the encoder
let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));

// 3. Build a stimulus
let input: Vec<f32> = (0..config.input_channels)
    .map(|i| i as f32 / (config.input_channels - 1) as f32)
    .collect();

// 4. Encode
let output = encoder.encode(&input);
println!("{}/{} channels fired", output.spikes.len(), config.input_channels);
```

## Design Philosophy

- **Standalone Abstraction**: Stripped of domain-specific logic to ensure it remains a pure encoding library.
- **Performance**: Minimal allocation in the hot path.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
