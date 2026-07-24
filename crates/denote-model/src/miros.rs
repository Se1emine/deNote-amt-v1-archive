//! Denote-MIROS-v1: MusicFM-style encoder + adapter + token decoder.

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
};
use denote_encoder::{MusicFmEncoder, MusicFmEncoderConfig};

use crate::adapter::{AdapterConfig, AdapterOutput, InstrumentAdapter};
use crate::decoder::{TokenDecoder, TokenDecoderConfig};

#[derive(Config, Debug)]
pub struct DenoteMirosConfig {
    pub encoder: MusicFmEncoderConfig,
    pub adapter: AdapterConfig,
    pub decoder: TokenDecoderConfig,
}

/// Full seq2seq AMT model.
#[derive(Module, Debug)]
pub struct DenoteMiros<B: Backend> {
    pub encoder: MusicFmEncoder<B>,
    pub adapter: InstrumentAdapter<B>,
    pub decoder: TokenDecoder<B>,
}

/// Forward outputs for training.
pub struct MirosOutput<B: Backend> {
    /// Token logits `[B, S, V]`.
    pub token_logits: Tensor<B, 3>,
    pub adapter: AdapterOutput<B>,
}

impl DenoteMirosConfig {
    /// Tiny CPU smoke configuration (not MusicFM-weight compatible).
    pub fn smoke(vocab_size: usize) -> Self {
        let enc = MusicFmEncoderConfig::smoke();
        let d = enc.encoder_dim;
        Self {
            encoder: enc,
            adapter: AdapterConfig::new(d, 8)
                .with_n_pitches_aux(128)
                .with_dropout(0.0),
            decoder: TokenDecoderConfig::new(vocab_size, d, 2, 4, d * 2).with_dropout(0.0),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DenoteMiros<B> {
        DenoteMiros {
            encoder: self.encoder.init(device),
            adapter: self.adapter.init(device),
            decoder: self.decoder.init(device),
        }
    }
}

impl<B: Backend> DenoteMiros<B> {
    /// Encode mel → adapter memory → decode tokens (teacher forcing).
    ///
    /// - `mel`: `[B, 1, n_mels, T]`
    /// - `tokens_in`: `[B, S]` (BOS + prefix)
    /// - `program_ids`: `[B]`
    pub fn forward(
        &self,
        mel: Tensor<B, 4>,
        tokens_in: Tensor<B, 2, Int>,
        program_ids: Tensor<B, 1, Int>,
    ) -> MirosOutput<B> {
        let latents = self.encoder.forward_mel(mel);
        self.forward_from_latents(latents, tokens_in, program_ids)
    }

    /// Skip encoder: latents already on device (latent-cache / L0 offline path).
    ///
    /// - `latents`: `[B, T, D]`
    /// - `tokens_in`: `[B, S]`
    /// - `program_ids`: `[B]`
    pub fn forward_from_latents(
        &self,
        latents: Tensor<B, 3>,
        tokens_in: Tensor<B, 2, Int>,
        program_ids: Tensor<B, 1, Int>,
    ) -> MirosOutput<B> {
        let adapter = self.adapter.forward(latents, program_ids);
        let token_logits = self
            .decoder
            .forward(tokens_in, adapter.memory.clone());
        MirosOutput {
            token_logits,
            adapter,
        }
    }

    /// Encode only (for AR inference loop).
    pub fn encode_memory(
        &self,
        mel: Tensor<B, 4>,
        program_ids: Tensor<B, 1, Int>,
    ) -> AdapterOutput<B> {
        let latents = self.encoder.forward_mel(mel);
        self.adapter.forward(latents, program_ids)
    }
}
