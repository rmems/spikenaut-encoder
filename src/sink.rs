//! Caller-owned destinations for emitted spikes.
//!
//! [`Encoder::encode`](crate::Encoder::encode) hands back an owned
//! [`EncodedOutput`], which means a fresh `Vec<SpikeEvent>` allocation on every
//! call. That is the right default for exploration and for one-shot encoding,
//! but a runtime that steps thousands of channels per millisecond wants the
//! opposite: one buffer, allocated once, refilled forever.
//!
//! [`SpikeSink`] is that seam. It is a single-method, object-safe trait for
//! "somewhere spikes can be written", and
//! [`Encoder::encode_into`](crate::Encoder::encode_into) /
//! [`Encoder::encode_step_into`](crate::Encoder::encode_step_into) write
//! through it instead of allocating. The crate ships adapters for
//! `Vec<SpikeEvent>` and [`EncodedOutput`]; downstream event buffers, ring
//! queues, hardware FIFOs, and format translators implement the trait
//! themselves and never materialize a `Vec` at all.
//!
//! ```rust
//! use axon_encoder::prelude::*;
//! # fn main() -> Result<(), EncoderError> {
//! let mut encoder = DeltaEncoder::try_new(0.1, 3)?;
//! let mut buffer: Vec<SpikeEvent> = Vec::new();
//!
//! for step in [[0.5, 0.0, 0.0], [0.5, 0.9, 0.0]] {
//!     buffer.clear(); // keeps the capacity, drops last step's contents
//!     encoder.encode_step_into(&step, &mut buffer);
//!     // ... drain `buffer` into the downstream runtime ...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`EncodedOutput`]: crate::types::EncodedOutput

use crate::types::{EncodedOutput, SpikeEvent};

/// A destination for spikes emitted by an encoder.
///
/// Implement this to receive spikes directly in your own representation —
/// a reusable buffer, a ring queue, a hardware event packet — instead of
/// taking the `Vec<SpikeEvent>` an [`Encoder`](crate::Encoder) would otherwise
/// allocate.
///
/// # Contract
///
/// - Encoders **append**. They never clear, truncate, or reorder what a sink
///   already holds, so a caller that wants one step per buffer clears it
///   between calls.
/// - Spikes arrive in the same order the returning APIs would place them in
///   [`EncodedOutput::spikes`], with identical channels, [`TickOffset`]s, and
///   polarities. Offsets stay call-relative; see the [`time`](crate::time)
///   module.
/// - Only spikes travel through a sink. `EncodedOutput`'s `embeddings` and
///   `metadata` are not produced by any encoder in this crate; a caller that
///   needs them uses the returning APIs.
///
/// The trait is object-safe on purpose: `&mut dyn SpikeSink` is what the
/// `encode_*_into` methods take, so `dyn Encoder` and `dyn ModulatedEncoder`
/// keep working.
///
/// [`TickOffset`]: crate::time::TickOffset
/// [`EncodedOutput::spikes`]: crate::types::EncodedOutput::spikes
///
/// # Examples
///
/// A sink that translates each spike into a caller-native pair as it arrives,
/// so no intermediate `Vec<SpikeEvent>` is ever built:
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// /// A downstream runtime's own event buffer.
/// struct EventQueue {
///     events: Vec<(u16, u64)>,
/// }
///
/// impl SpikeSink for EventQueue {
///     fn push(&mut self, event: SpikeEvent) {
///         self.events.push((event.channel, event.timestamp.ticks()));
///     }
///
///     fn reserve(&mut self, additional: usize) {
///         self.events.reserve(additional);
///     }
/// }
///
/// let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
/// let mut queue = EventQueue { events: Vec::new() };
///
/// encoder.encode_step_into(&[1.0, 0.0], &mut queue);
/// // Strong input spikes at tick 0, weak input at the end of the window.
/// assert_eq!(queue.events, vec![(0, 0), (1, 9)]);
/// # Ok(())
/// # }
/// ```
pub trait SpikeSink {
    /// Accepts one spike.
    fn push(&mut self, event: SpikeEvent);

    /// Hints that `additional` more spikes are about to be pushed.
    ///
    /// Buffer-backed sinks override this to pre-size and avoid repeated growth.
    /// It is only a hint: encoders may push more or fewer spikes than they
    /// reserved, and the default implementation does nothing.
    fn reserve(&mut self, additional: usize) {
        let _ = additional;
    }
}

/// The canonical reusable buffer: clear it between steps and the capacity stays.
impl SpikeSink for Vec<SpikeEvent> {
    #[inline]
    fn push(&mut self, event: SpikeEvent) {
        Vec::push(self, event);
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }
}

/// Appends into [`EncodedOutput::spikes`], leaving the other fields untouched.
///
/// [`EncodedOutput::spikes`]: crate::types::EncodedOutput::spikes
impl SpikeSink for EncodedOutput {
    #[inline]
    fn push(&mut self, event: SpikeEvent) {
        self.spikes.push(event);
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.spikes.reserve(additional);
    }
}

/// Forwards through a mutable borrow, so wrapper sinks can hold `&mut dyn SpikeSink`.
impl<S: SpikeSink + ?Sized> SpikeSink for &mut S {
    #[inline]
    fn push(&mut self, event: SpikeEvent) {
        (**self).push(event);
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        (**self).reserve(additional);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TickOffset;

    fn spike(channel: u16) -> SpikeEvent {
        SpikeEvent::at_step_start(channel, true)
    }

    #[test]
    fn vec_sink_appends_and_reserves() {
        let mut sink: Vec<SpikeEvent> = Vec::new();
        SpikeSink::reserve(&mut sink, 4);
        assert!(sink.capacity() >= 4);

        SpikeSink::push(&mut sink, spike(0));
        SpikeSink::push(&mut sink, spike(1));
        assert_eq!(sink, vec![spike(0), spike(1)]);
    }

    #[test]
    fn encoded_output_sink_only_touches_spikes() {
        let mut output = EncodedOutput::new();
        SpikeSink::reserve(&mut output, 2);
        assert!(output.spikes.capacity() >= 2);

        SpikeSink::push(&mut output, spike(7));
        assert_eq!(output.spikes, vec![spike(7)]);
        assert!(output.embeddings.is_none());
        assert!(output.metadata.is_none());
    }

    #[test]
    fn mutable_borrow_forwards_to_the_inner_sink() {
        let mut buffer: Vec<SpikeEvent> = Vec::new();

        {
            // The blanket impl is what lets a wrapper hold `&mut dyn SpikeSink`.
            let borrowed: &mut dyn SpikeSink = &mut buffer;
            borrowed.reserve(1);
            borrowed.push(SpikeEvent::new(3, TickOffset::new(9), false));
        }

        assert_eq!(buffer, vec![SpikeEvent::new(3, 9u64, false)]);
    }

    #[test]
    fn default_reserve_is_a_no_op() {
        struct CountOnly(usize);
        impl SpikeSink for CountOnly {
            fn push(&mut self, _event: SpikeEvent) {
                self.0 += 1;
            }
        }

        let mut sink = CountOnly(0);
        sink.reserve(1024); // inherited default: must not panic or allocate
        sink.push(spike(0));
        assert_eq!(sink.0, 1);
    }

    #[test]
    fn sinks_are_object_safe() {
        let mut buffer: Vec<SpikeEvent> = Vec::new();
        let sink: &mut dyn SpikeSink = &mut buffer;
        sink.push(spike(1));
        assert_eq!(buffer.len(), 1);
    }
}
