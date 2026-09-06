# Axon Encoder

[![CI](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Limen-Neural/axon-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/axon-encoder)
[![Qodana](https://github.com/Limen-Neural/axon-encoder/actions/workflows/qodana_code_quality.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/qodana_code_quality.yml)
[![Docs](https://docs.rs/axon-encoder/badge.svg)](https://docs.rs/axon-encoder)

**A flexible sensory encoding library for spiking neural networks (SNNs).**

`axon-encoder` turns continuous data—sensor readings, telemetry, control
signals—into **spikes**, the event-based signals SNNs process. Use it as the
front-end of a neuromorphic pipeline without pulling in a full SNN simulator.

## Installation

**0.4.x is experimental (pre-1.0).** Cargo treats `axon-encoder = "0.4"` as
`^0.4` (that is `>= 0.4.0, < 0.5.0`): compatible **patch** updates only.
A `0.5` release is a new breaking line; pin `"=0.4.0"` if you need an exact
crate version.

```toml
[dependencies]
axon-encoder = "0.4"
```

Optional features:

| Feature | Purpose |
| --- | --- |
| `serde` | Serialize configs and gain types |
| `ndarray` | Encode from `ndarray` views (`ArrayView1` / `ArrayView2`) |

```toml
[dependencies]
axon-encoder = { version = "0.4", features = ["ndarray"] }
ndarray = "0.16" # declare yourself so you can build ArrayView values
```

Requires **Rust 1.97.1+** (edition 2024). See `rust-version` in `Cargo.toml`.

## Quick start

```rust
use axon_encoder::prelude::*;

fn main() {
    // Prefer try_new: typed validation instead of panics on bad config.
    // Range is (min, max); values are clamped to that span. Endpoints map to
    // base_rate / max_rate (here 5–100 Hz at a 10 ms sampling interval).
    let mut encoder = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010)
        .expect("valid RateEncoder configuration");

    // Inclusive endpoints 0.0 ..= 1.0 (matches the range above).
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
    let output = encoder.encode(&input);

    println!(
        "Input of {} values produced {} spikes.",
        input.len(),
        output.spikes.len()
    );
}
```

Full API docs: [docs.rs/axon-encoder](https://docs.rs/axon-encoder).

## Reusing storage: `encode_into` and `SpikeSink`

`encode` / `encode_step` allocate a fresh `Vec<SpikeEvent>` per call. That is
the right default for exploration, but a runtime stepping thousands of channels
wants one buffer, allocated once. `encode_into` / `encode_step_into` write into
any `SpikeSink` you own instead:

```rust
use axon_encoder::prelude::*;

fn main() {
    let mut encoder = DeltaEncoder::try_new(0.1, 3).expect("valid DeltaEncoder");
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for step in [[0.5, 0.0, 0.0], [0.5, 0.9, 0.0]] {
        buffer.clear(); // keeps the capacity, drops last step's spikes
        encoder.encode_step_into(&step, &mut buffer);
        println!("{} spikes", buffer.len());
    }
}
```

Same spikes, same order, same state advancement as the returning APIs — and
zero allocations per step once the buffer is warm. `Vec<SpikeEvent>` and
`EncodedOutput` implement `SpikeSink` out of the box; a downstream event
buffer, ring queue, or hardware adapter implements the one-method trait itself
and never materializes a `Vec` at all:

```rust
use axon_encoder::prelude::*;

struct EventQueue {
    events: Vec<(u16, u64)>,
}

impl SpikeSink for EventQueue {
    fn push(&mut self, event: SpikeEvent) {
        self.events.push((event.channel, event.timestamp.ticks()));
    }

    fn reserve(&mut self, additional: usize) {
        self.events.reserve(additional);
    }
}
```

`SpikeSink` has a third, optional method — `extend_from_slice` — which defaults
to `push` in a loop. Override it when your sink can take a slice more cheaply
than repeated pushes; encoders deliver spikes through it in fixed-size runs, so
writing through a trait object costs one virtual call per run rather than per
spike.

Encoders **append** to a sink and never clear it, so the caller decides where
step boundaries are. The trait is object-safe, so `&mut dyn Encoder` and
`&mut dyn ModulatedEncoder` still work; `ModulatedEncoder` has the matching
`encode_with_gains_into` / `encode_with_modulators_into`. `PoissonEncoder` and
`EmbeddingRateEncoder` are documented exceptions — neither implements
`Encoder`. See `cargo run --example encode_into_sink`.

## Spike time semantics

Every encoder here shares **one time model**, so a consumer can integrate any of
them — or a `&mut dyn Encoder` — without special cases:

> A `SpikeEvent::timestamp` is a `TickOffset`: a count of encoder **ticks**
> measured from the start of the `encode` / `encode_step` call that emitted it.

Timestamps are **call-relative**. They are never absolute and never wall-clock,
because this crate owns no clock and no scheduler. The caller keeps absolute
time in a `TimeCursor` and advances it once per call:

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
    let mut cursor = TimeCursor::new(encoder.time_model());

    for _ in 0..3 {
        let output = encoder.encode_step(&[0.9, 0.1]);
        for spike in &output.spikes {
            // Absolute tick on your timeline; absolute_nanos(..) when a
            // Timebase is available.
            let _tick = cursor.absolute(spike.timestamp);
        }
        cursor.advance(); // by time_model().step_ticks()
    }
    Ok(())
}
```

`Encoder::time_model()` reports the three things a consumer needs:

| | Meaning |
| --- | --- |
| `step_ticks()` | How far your origin advances per call |
| `span_ticks()` | Exclusive bound on offsets a single call can emit |
| `timebase()` | Physical duration of one tick, when the encoder knows it |

Per encoder:

| Encoder | `step_ticks` | `span_ticks` | `timebase` |
| --- | --- | --- | --- |
| `RateEncoder` | 1 | 1 | `dt_seconds` |
| `LatencyEncoder` | `max_latency + 1` | `max_latency + 1` | none |
| `PhaseEncoder` | 1 | `cycle_steps` | none |
| `PopulationEncoder`, `DeltaEncoder`, `DerivativeEncoder`, `TemporalEncoder`, `PredictiveEncoder` | 1 | 1 | none |

**Batch versus streaming.** Both modes follow the same rule, once per call —
`encode` is not a longer window than `encode_step`. `PhaseEncoder` advances its
oscillation by one tick in either mode; the stateful encoders update history in
either mode; `LatencyEncoder` is stateless, so the two are identical.

**Ordering.** Within one `spikes` slice: channel IDs are non-decreasing, offsets
are non-decreasing within a channel, and repeated spikes from one channel at one
offset (a `RateEncoder` burst) are contiguous and mutually unordered — the run
length is a spike *count*, not a sequence. Note this is channel-major, not
globally time-sorted: sort by `timestamp` if you need a chronological stream.

`PhaseEncoder` is the one encoder whose calls overlap (`span_ticks >
step_ticks`), since a call can place a spike anywhere in the ongoing cycle.

Run `cargo run --example spike_timebase` for a worked integration: two encoders,
two cursors, one merged nanosecond-timed stream.

### Migrating from 0.4

Two breaking changes, both in the 0.5 line:

1. **`SpikeEvent::timestamp` is now `TickOffset`, not `u64`.** The type converts
   both ways and compares against `u64`, so reads like
   `assert_eq!(spike.timestamp, 5)` and `spike.timestamp <= other` still work.
   Construction sites need `SpikeEvent::new(channel, 5u64, true)`,
   `SpikeEvent::at_step_start(channel, true)`, or `TickOffset::new(5)` in the
   struct literal; use `spike.timestamp.ticks()` where a raw `u64` is required.
   The serde representation is unchanged — `TickOffset` is `#[serde(transparent)]`,
   so 0.4 payloads still deserialize.
2. **`PhaseEncoder` emits call-relative offsets.** It previously emitted
   `current_phase + phase_offset`, an absolute value that no other encoder used.
   The old number is `cursor.absolute(spike.timestamp)`, or
   `phase_before_the_call + spike.timestamp.ticks()` if you track the
   oscillation yourself; cycle position stays `absolute % cycle_steps`.
   Capture `current_phase()` **before** the emitting call — every encode call
   advances it afterward, so a read taken after the call is one tick ahead.

Two smaller behavior changes, both in service of making `span_ticks()` a hard
bound rather than an advisory one:

- A neuromodulated `latency_scale` above `1.0` no longer stretches spikes past
  `max_latency`. Latency gains still shorten the window.
- `LatencyEncoder::try_new` rejects `max_latency == u64::MAX` with
  `EncoderError::WindowTooLarge`, since the presentation window is
  `max_latency + 1` ticks and that value has no representable window. The
  `serde` `Deserialize` impl routes through `try_new`, so a 0.4 payload
  persisted with that `max_latency` now fails to load rather than round-tripping.

`Encoder::time_model()` has a default implementation, so out-of-crate `Encoder`
impls keep compiling and inherit `TimeModel::INSTANT`.

### Rate encoder time semantics

`RateEncoder` treats `base_rate` and `max_rate` as firing rates in **hertz**.
Prefer `RateEncoder::try_new(base_rate_hz, max_rate_hz, range, dt_seconds)` so the
sampling interval is explicit (finite and strictly positive). Stochastic batch
encoding uses `p = 1 - exp(-rate_hz * dt_seconds)`; streaming accumulates
`phase += rate_hz * dt_seconds`.

`RateEncoder::new(base_rate, max_rate, range)` remains for compatibility and
uses `dt_seconds = 0.1`.

### Constructor errors

Most encoders expose `try_new(...) -> Result<Self, EncoderError>` for invalid
rates, ranges, windows, thresholds, or channel counts. Prefer those over
panicking `new(...)` in libraries and applications. `PredictiveEncoder` is the
exception: its `new(...)` already returns a `Result`.

## Features

- **Encoders** for different signal structures:
  - **`RateEncoder`** — spike *rate* tracks input magnitude
  - **`DerivativeEncoder`** — fires on *change* (jumps / drops)
  - **`TemporalEncoder`** — *patterns* over time
  - **`PopulationEncoder`** — value distributed across a *population* of units
  - **`DeltaEncoder`** — spike when the signal moves by a threshold
  - **`LatencyEncoder`** — stronger input → earlier spike in a window
  - **`PoissonEncoder`** — Poisson-process style sampling
- **`Encoder` / `ModulatedEncoder` traits** — plug in custom encoders or apply
  gain scales (`EncodingGains`) without owning a full neuromodulator runtime
- **`SpikeSink` + `encode_into`** — write spikes into caller-owned storage and
  reuse one buffer across steps, or translate straight into your own event type
- **Optional `ndarray` helpers** — `NdarrayEncoderExt` for view-based batch input
- **Small dependency surface** — easy to embed in larger systems

## Randomness (stochastic encoders)

`RateEncoder`, `PopulationEncoder`, and `PoissonEncoder` sample unit floats in
`[0, 1)` via `axon_encoder::rng`:

- **Default:** `gen_unit_f32()` uses a thread-local `rand` generator (not
  reproducible across runs).
- **Reproducible runs:** `gen_unit_f32_with_rng(&mut rng)` with a seeded RNG
  (for example `rand::rngs::StdRng`).
- For **encoding only** — not cryptographic use.

## WebAssembly

On `wasm32-unknown-unknown`, enable a working
[getrandom](https://docs.rs/getrandom) backend for your target (often the
JS/browser feature set). Stochastic encoders need OS/entropy-backed RNGs
through `rand`.

## Examples

Clone the repository and run:

```bash
cargo run --example rate_encoding
cargo run --example delta_encoding
cargo run --example spike_timebase
cargo run --example encode_into_sink
cargo run --example ndarray_encoding --features ndarray
```

Other examples live under `examples/` (latency, population, temporal,
predictive, gain-adapter patterns, and more).

## What this crate is (and is not)

### In scope

- Sensory / signal → spike encoding algorithms
- Deterministic and stochastic encoding pipelines
- Generic gain controls (`EncodingGains`, gain curves) used only for scaling
  rate, threshold, latency, or sensitivity at encode time

### Out of scope

- Full SNN simulation, network topology, or synaptic plasticity (STDP)
- Long-horizon biological neuromodulator *dynamics* or reward loops (this crate
  only provides encoding-local gain helpers)
- FPGA / ASIC / GPU device bindings

The library is intentionally unopinionated about which simulator or hardware
stack you plug the spikes into.

## Docker (optional)

Published images ship **example binaries** (not a substitute for depending on
the crate from Cargo):

```bash
docker pull ghcr.io/limen-neural/axon-encoder:0.4.0
docker run --rm ghcr.io/limen-neural/axon-encoder:0.4.0
```

Build locally from a git checkout:

```bash
docker build -t axon-encoder:dev .
docker run --rm axon-encoder:dev

docker build --target builder -t axon-encoder:builder .
docker run --rm axon-encoder:builder   # cargo test --all-features --locked
```

## Contributing

Issues and pull requests are welcome—new encoders, fixes, and docs improvements
alike. Development notes and CI conventions live in the repository
(`REVIEW.md`, `.github/`).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
