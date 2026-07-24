//! Modular multi-decoder container (Phase 2).
//!
//! Job B_g trains **one** `decoder_g` with frozen adapter/presence from Job A.
//! Infer loads all group decoders and presence-gates inactive groups.

use std::path::Path;

use burn::{
    config::Config,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::Backend, Int, Tensor},
};

use crate::adapter::{AdapterConfig, AdapterOutput, InstrumentAdapter};
use crate::decoder::{TokenDecoder, TokenDecoderConfig};
use crate::presence::{PresenceHead, PresenceHeadConfig};

/// Config for modular system (shared decoder hyperparams × G).
#[derive(Config, Debug)]
pub struct DenoteModularConfig {
    pub adapter: AdapterConfig,
    pub presence: PresenceHeadConfig,
    pub decoder: TokenDecoderConfig,
    /// Number of instrument groups G.
    pub n_groups: usize,
}

/// Container: frozen Job A modules + G independent token decoders.
///
/// Not a single Burn `Module` over all parts — Job B optimizes only `decoders[g]`.
#[derive(Debug)]
pub struct DenoteModular<B: Backend> {
    pub adapter: InstrumentAdapter<B>,
    pub presence: PresenceHead<B>,
    pub decoders: Vec<TokenDecoder<B>>,
    pub d_model: usize,
    pub n_programs: usize,
    pub vocab_size: usize,
}

/// Teacher-forcing outputs for one group.
pub struct ModularGroupOutput<B: Backend> {
    pub token_logits: Tensor<B, 3>,
    pub adapter: AdapterOutput<B>,
}

impl DenoteModularConfig {
    pub fn smoke(vocab_size: usize, d_model: usize, n_programs: usize, n_groups: usize) -> Self {
        Self {
            adapter: AdapterConfig::new(d_model, n_programs)
                .with_n_pitches_aux(128)
                .with_dropout(0.0),
            presence: PresenceHeadConfig::new(d_model, n_programs).with_hidden(d_model),
            decoder: TokenDecoderConfig::new(vocab_size, d_model, 2, 4, d_model * 2).with_dropout(0.0),
            n_groups: n_groups.max(1),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DenoteModular<B> {
        let n = self.n_groups.max(1);
        let decoders = (0..n).map(|_| self.decoder.init(device)).collect();
        DenoteModular {
            adapter: self.adapter.init(device),
            presence: self.presence.init(device),
            decoders,
            d_model: self.adapter.d_model,
            n_programs: self.presence.n_programs,
            vocab_size: self.decoder.vocab_size,
        }
    }
}

impl<B: Backend> DenoteModular<B> {
    pub fn n_groups(&self) -> usize {
        self.decoders.len()
    }

    /// Load adapter + presence from Job A modular checkpoint directory
    /// (expects `adapter.mpk` + `presence.mpk` stems).
    pub fn load_job_a(
        &mut self,
        job_a_dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<(), String> {
        let dir = job_a_dir.as_ref();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let adapter_stem = dir.join("adapter");
        let presence_stem = dir.join("presence");
        if !stem_mpk_exists(&adapter_stem) {
            return Err(format!(
                "Job A adapter missing at {}.mpk",
                adapter_stem.display()
            ));
        }
        self.adapter = self
            .adapter
            .clone()
            .load_file(adapter_stem.to_string_lossy().as_ref(), &recorder, device)
            .map_err(|e| format!("load adapter: {e}"))?;
        self.presence = self
            .presence
            .clone()
            .load_file(presence_stem.to_string_lossy().as_ref(), &recorder, device)
            .map_err(|e| format!("load presence: {e}"))?;
        Ok(())
    }

    pub fn load_decoder_g(
        &mut self,
        g: usize,
        stem: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<(), String> {
        if g >= self.decoders.len() {
            return Err(format!("group {g} out of range (n={})", self.decoders.len()));
        }
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let stem = stem.as_ref();
        self.decoders[g] = self.decoders[g]
            .clone()
            .load_file(stem.to_string_lossy().as_ref(), &recorder, device)
            .map_err(|e| format!("load decoder_g{g}: {e}"))?;
        Ok(())
    }

    pub fn save_decoder_g(&self, g: usize, stem: impl AsRef<Path>) -> Result<(), String> {
        if g >= self.decoders.len() {
            return Err(format!("group {g} out of range (n={})", self.decoders.len()));
        }
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        self.decoders[g]
            .clone()
            .save_file(stem.as_ref().to_string_lossy().as_ref(), &recorder)
            .map_err(|e| format!("save decoder_g{g}: {e}"))?;
        Ok(())
    }

    /// Save adapter + presence (usually unchanged Job A copies) next to decoder.
    pub fn save_job_a_modules(&self, dir: impl AsRef<Path>) -> Result<(), String> {
        let dir = dir.as_ref();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        self.adapter
            .clone()
            .save_file(dir.join("adapter").to_string_lossy().as_ref(), &recorder)
            .map_err(|e| format!("save adapter: {e}"))?;
        self.presence
            .clone()
            .save_file(dir.join("presence").to_string_lossy().as_ref(), &recorder)
            .map_err(|e| format!("save presence: {e}"))?;
        Ok(())
    }

    /// Teacher-forcing forward for group `g`.
    ///
    /// - `z`: `[B, T, D]` latents
    /// - `tokens_in`: `[B, S]` (BOS + prefix)
    /// - `program_multi_hot`: `[B, K]` active programs for conditioning
    /// - `detach_adapter`: if true, freeze adapter (Job B default)
    pub fn forward_group_tf(
        &self,
        g: usize,
        z: Tensor<B, 3>,
        tokens_in: Tensor<B, 2, Int>,
        program_multi_hot: Tensor<B, 2>,
        detach_adapter: bool,
    ) -> ModularGroupOutput<B> {
        assert!(g < self.decoders.len(), "group {g} out of range");
        let adapter = self.adapter.forward_multi_hot(z, program_multi_hot);
        let memory = if detach_adapter {
            adapter.memory.clone().detach()
        } else {
            adapter.memory.clone()
        };
        let token_logits = self.decoders[g].forward(tokens_in, memory);
        ModularGroupOutput {
            token_logits,
            adapter,
        }
    }

    /// Presence logits from frozen adapter (infer / gate).
    ///
    /// Uses all-zero multi-hot for neutral program conditioning when probing presence
    /// (presence is mean-pooled over adapted memory; Job A trained with clip multi-hot —
    /// at infer we typically pass clip multi-hot from GT or zeros + re-run).
    pub fn presence_from_latents(
        &self,
        z: Tensor<B, 3>,
        program_multi_hot: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let adapter = self.adapter.forward_multi_hot(z, program_multi_hot);
        self.presence.forward(adapter.memory)
    }

    /// Adapter memory only (for AR decode), optionally detached.
    pub fn adapter_memory(
        &self,
        z: Tensor<B, 3>,
        program_multi_hot: Tensor<B, 2>,
        detach: bool,
    ) -> Tensor<B, 3> {
        let adapter = self.adapter.forward_multi_hot(z, program_multi_hot);
        if detach {
            adapter.memory.detach()
        } else {
            adapter.memory
        }
    }

    pub fn decoder(&self, g: usize) -> &TokenDecoder<B> {
        &self.decoders[g]
    }

    pub fn decoder_mut(&mut self, g: usize) -> &mut TokenDecoder<B> {
        &mut self.decoders[g]
    }
}

fn stem_mpk_exists(stem: &Path) -> bool {
    stem.with_extension("mpk").exists()
        || Path::new(&format!("{}.mpk", stem.display())).exists()
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Int;

    use super::*;

    type B = NdArray;

    #[test]
    fn modular_forward_shapes() {
        let device = Default::default();
        let vocab = 4 + 205 + 128 + 128 + 128 + 8;
        let d = 32usize;
        let k = 8usize;
        let g = 4usize;
        let model = DenoteModularConfig::smoke(vocab, d, k, g).init::<B>(&device);
        assert_eq!(model.n_groups(), 4);
        let z = Tensor::<B, 3>::zeros([2, 10, d], &device);
        let tokens = Tensor::<B, 2, Int>::zeros([2, 12], &device);
        let multi = Tensor::<B, 2>::zeros([2, k], &device);
        let out = model.forward_group_tf(0, z, tokens, multi, true);
        assert_eq!(out.token_logits.dims(), [2, 12, vocab]);
    }

    #[test]
    fn only_decoder_params_change_concept() {
        // Smoke: detach path returns finite logits; optim wiring is Job B's responsibility.
        let device = Default::default();
        let model = DenoteModularConfig::smoke(100, 16, 4, 2).init::<B>(&device);
        let z = Tensor::<B, 3>::zeros([1, 4, 16], &device);
        let tokens = Tensor::<B, 2, Int>::zeros([1, 5], &device);
        let multi = Tensor::<B, 2>::zeros([1, 4], &device);
        let out = model.forward_group_tf(1, z, tokens, multi, true);
        assert_eq!(out.token_logits.dims()[2], 100);
    }
}
