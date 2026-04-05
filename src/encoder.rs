use candle_core::{Tensor, Result, Device, Shape, DType};

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub v_th: f64, 
}

#[derive(Clone)]
pub struct EncoderState {
    pub u_enc: Tensor, 
}

impl EncoderState {
    pub fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            u_enc: Tensor::zeros(shape, dtype, device)?,
        })
    }
}

pub struct EmbeddingRateEncoder {
    pub config: EncoderConfig,
    pub normalized_embeddings: Tensor, 
}

impl EmbeddingRateEncoder {
    /// Ingests OLMo embeddings and applies Linear Min-Max Scaling
    /// to preserve spatial relationships between tokens.
    pub fn new(embeddings: &Tensor, config: EncoderConfig) -> Result<Self> {
        // Linear Min-Max Scaling: (X - X_min) / (X_max - X_min)
        // We calculate this along the embedding dimension (dim 2 usually: batch, seq, hidden_dim)
        let dim = embeddings.rank() - 1; 
        
        let min_val = embeddings.min_keepdim(dim)?;
        let max_val = embeddings.max_keepdim(dim)?;
        
        let range = max_val.sub(&min_val)?;
        
        // Add a tiny epsilon to prevent division by zero in dead neurons
        let epsilon = Tensor::new(1e-5f32, embeddings.device())?
                        .to_dtype(embeddings.dtype())?
                        .broadcast_as(range.shape())?;
        let safe_range = range.add(&epsilon)?;

        let centered = embeddings.sub(&min_val)?;
        let normalized = centered.broadcast_div(&safe_range)?;

        Ok(Self {
            config,
            normalized_embeddings: normalized,
        })
    }

    /// The highly optimized forward step. 
    /// Note: In production, this block will be replaced by a custom `candle_core::CustomOp1` 
    /// to fuse the add, compare, and subtract into a single C++ CUDA pass.
    pub fn forward_t(&self, prev_state: &EncoderState) -> Result<(Tensor, EncoderState)> {
        // Accumulate current
        let updated_u = prev_state.u_enc.add(&self.normalized_embeddings)?;

        // Evaluate spikes
        let threshold = Tensor::new(self.config.v_th, updated_u.device())?
                                .to_dtype(updated_u.dtype())?
                                .broadcast_as(updated_u.shape())?;
        
        // We keep spikes as a minimal datatype (u8 equivalent) instead of casting back to f16
        let spikes = updated_u.ge(&threshold)?; 
        
        // Soft reset: Only subtract threshold where spikes == 1
        // We temporarily cast the mask to f16/f32 just for the subtraction math
        let spikes_float = spikes.to_dtype(updated_u.dtype())?;
        let reset_subtraction = spikes_float.broadcast_mul(&threshold)?;
        let new_u_enc = updated_u.sub(&reset_subtraction)?;

        // Return the boolean/u8 spikes and the updated state
        Ok((spikes_float, EncoderState { u_enc: new_u_enc }))
    }
}
