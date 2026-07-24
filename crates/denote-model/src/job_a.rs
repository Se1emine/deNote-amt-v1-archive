//! Job A model: adapter + presence, **no encoder**.
//!
//! Train on latent cache only — encoder weights are never loaded into VRAM.
//! Mode A2 (token CE decoder) is optional Phase 1 stretch; primary path is
//! presence + adapter only (Mode A1).

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
};

use crate::adapter::{AdapterConfig, AdapterOutput, InstrumentAdapter};
use crate::presence::{PresenceHead, PresenceHeadConfig};

#[derive(Config, Debug)]
pub struct DenoteJobAConfig {
    pub adapter: AdapterConfig,
    pub presence: PresenceHeadConfig,
}

/// Trainable Job A modules (encoder not present).
#[derive(Module, Debug)]
pub struct DenoteJobA<B: Backend> {
    pub adapter: InstrumentAdapter<B>,
    pub presence: PresenceHead<B>,
}

/// Forward outputs for Job A training / val.
pub struct JobAOutput<B: Backend> {
    pub adapter: AdapterOutput<B>,
    /// Presence logits `[B, K]`.
    pub presence_logits: Tensor<B, 2>,
}

impl DenoteJobAConfig {
    /// Smoke-scale Job A (matches encoder smoke `d_model`).
    pub fn smoke(d_model: usize, n_programs: usize) -> Self {
        Self {
            adapter: AdapterConfig::new(d_model, n_programs)
                .with_n_pitches_aux(128)
                .with_dropout(0.0),
            presence: PresenceHeadConfig::new(d_model, n_programs).with_hidden(d_model),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DenoteJobA<B> {
        DenoteJobA {
            adapter: self.adapter.init(device),
            presence: self.presence.init(device),
        }
    }
}

impl<B: Backend> DenoteJobA<B> {
    /// Forward from precomputed latents (Job A path — no encoder).
    ///
    /// - `z`: `[B, T, D]`
    /// - `program_multi_hot`: `[B, K]` floats 0/1 (sum of active program embeddings)
    pub fn forward_from_latents(
        &self,
        z: Tensor<B, 3>,
        program_multi_hot: Tensor<B, 2>,
    ) -> JobAOutput<B> {
        let adapter = self.adapter.forward_multi_hot(z, program_multi_hot);
        let presence_logits = self.presence.forward(adapter.memory.clone());
        JobAOutput {
            adapter,
            presence_logits,
        }
    }

    /// Single-program convenience (program id per batch).
    pub fn forward_from_latents_program_ids(
        &self,
        z: Tensor<B, 3>,
        program_ids: Tensor<B, 1, Int>,
    ) -> JobAOutput<B> {
        let adapter = self.adapter.forward(z, program_ids);
        let presence_logits = self.presence.forward(adapter.memory.clone());
        JobAOutput {
            adapter,
            presence_logits,
        }
    }
}
