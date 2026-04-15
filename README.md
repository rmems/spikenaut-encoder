# spikenaut-encoder

**Warp-Optimized 1024-Channel sensory encoding for spiking neural networks, accelerated via myelin-accelerator.**

`spikenaut-encoder` converts continuous experimental telemetry and time-series data into biologically plausible spike trains. It serves as the front-end sensory layer for the Spikenaut ecosystem — built to scale to 1024+ input channels in a single CUDA warp dispatch on the RTX 5080 Blackwell architecture.

## Features

- **GPU Acceleration**: Utilizes `myelin-accelerator` and CUDA kernels for massively parallel Poisson spike generation (RTX 5080 Blackwell optimized).
- **1024-Channel Baseline**: `EncoderConfig::default()` sets `input_channels = 1024` — fully saturating a CUDA warp grid in a single kernel dispatch.
- **Core Encoders**: Rate, Temporal, Predictive (anomaly), Population, and Neuromodulator-driven strategies.
- **Pure Biological Modulators**: Neuromodulation logic (Dopamine, Cortisol, Acetylcholine, Tempo) stripped of domain-specific noise for universal application.
- **Lightweight Tensors**: Built on the `corinth-canal` backbone for optimized memory throughput and minimal overhead.
- **Standardized Output**: `EncodedOutput` providing discrete `SpikeEvent` sequences and optional high-dimensional embeddings for hybrid SNN-LLM fusion.

## Installation

```toml
[dependencies]
spikenaut-encoder = { path = "../spikenaut-encoder" }
myelin-accelerator = { path = "../myelin-accelerator" }
corinth-canal = { path = "../corinth-canal" }
```

## Quick Start (requires GPU context)

```rust
use spikenaut_encoder::prelude::*;
use myelin_accelerator::GpuAccelerator;

// 1. Initialize GPU context (owned by parent application — never constructed internally)
let gpu = GpuAccelerator::new();

// 2. Load the Blackwell 1024-channel baseline config
let config = EncoderConfig::default(); // input_channels: 1024, output_channels: 1024

// 3. Initialize the encoder
let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));

// 4. Build a 1024-element stimulus — saturates a full CUDA warp grid in one dispatch
let input: Vec<f32> = (0..config.input_channels)
    .map(|i| i as f32 / (config.input_channels - 1) as f32)
    .collect();

// 5. Encode (GPU-accelerated parallel Poisson generation on RTX 5080)
let output = encoder.encode(&input, &gpu);
println!("{}/{} channels fired", output.spikes.len(), config.input_channels);
```

## Neuromodulatory Sensory Logic

The `NeuromodSensoryEncoder` implements homeostatic adaptation and dynamic gain control influenced by biological modulator states:
- **Dopamine**: Increases gain/sensitivity.
- **Cortisol**: Modulates baseline channel biases.
- **Acetylcholine**: Enhances adaptation rates for heightened focus.
- **Tempo**: Direct scaling of firing probabilities.

## Design Philosophy

- **VRAM Context Safety**: The `Encoder` trait requires a borrowed `&GpuAccelerator`. This prevents multiple encoders from spawning duplicate CUDA contexts and crashing VRAM.
- **Standalone Abstraction**: Stripped of HFT, mining, or market-specific domain logic to ensure it remains a pure mathematical and biological encoding library.
- **Performance First**: Minimal allocation in the hot path, leveraging raw device buffers for spike generation wherever possible.

## Integration

`spikenaut-encoder` is designed to feed event-based data into other Spikenaut libraries like `spikenaut-synapse`. For hybrid workflows, its embedding outputs are compatible with `spike-lmo` fusion projectors.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
