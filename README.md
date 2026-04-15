# spikenaut-encoder

**High-performance sensory encoding for spiking neural networks, accelerated via myelin-accelerator.**

`spikenaut-encoder` converts continuous experimental telemetry and time-series data into biologically plausible spike trains. It serves as the front-end sensory layer for the Spikenaut ecosystem, providing high-fidelity, GPU-accelerated encoding strategies for neuromorphic research and cyber-physical systems.

## Features

- **GPU Acceleration**: Utilizes `myelin-accelerator` and CUDA kernels for massively parallel Poisson spike generation (RTX 5080 optimized).
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

// 1. Initialize GPU context (owned by parent application)
let gpu = GpuAccelerator::new();

// 2. Initialize a 2-channel rate encoder
let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 100.0));
let input = [75.0, 25.0];

// 3. Encode (GPU-accelerated Poisson generation)
let output = encoder.encode(&input, &gpu);

for spike in output.spikes {
    println!("Spike on channel {}, polarity: {}", spike.channel, spike.polarity);
}
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
