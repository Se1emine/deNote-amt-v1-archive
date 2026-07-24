//! Denote models for automatic music transcription.
//!
//! **Primary path (Denote-MIROS-v1):** pretrained-style audio encoder embeddings
//! + instrument adapter + seq2seq MIDI-token decoder.
//!
//! **Legacy path:** Denote-CRNN-v1 (Onsets-and-Frames style) behind `legacy-crnn`.

#[cfg(feature = "legacy-crnn")]
mod crnn;
#[cfg(feature = "legacy-crnn")]
mod loss;

mod adapter;
mod decoder;
mod groups;
mod job_a;
mod miros;
mod modular;
mod presence;
mod token_loss;

#[cfg(feature = "legacy-crnn")]
pub use crnn::{CrnnConfig, CrnnOutput, DenoteCrnn};
#[cfg(feature = "legacy-crnn")]
pub use loss::{multi_task_loss, LossWeights, MultiTaskLoss};

pub use adapter::{AdapterConfig, AdapterOutput, InstrumentAdapter};
pub use decoder::{TokenDecoder, TokenDecoderConfig};
pub use groups::{GroupDef, GroupsConfig};
pub use job_a::{DenoteJobA, DenoteJobAConfig, JobAOutput};
pub use miros::{DenoteMiros, DenoteMirosConfig, MirosOutput};
pub use modular::{DenoteModular, DenoteModularConfig, ModularGroupOutput};
pub use presence::{
    presence_bce_logits, presence_f1_from_slices, PresenceF1, PresenceHead, PresenceHeadConfig,
};
pub use token_loss::{miros_loss, soft_bce_logits, token_ce_loss, MirosLoss, MirosLossWeights};

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Int;
    use denote_encoder::MusicFmEncoderConfig;

    use super::*;

    type B = NdArray;

    #[test]
    fn miros_smoke_forward_shapes() {
        let device = Default::default();
        // ~MT3-like vocab size (specials + time + on/off + vel + programs)
        let vocab_size = 4 + 205 + 128 + 128 + 128 + 8;
        let model = DenoteMirosConfig::smoke(vocab_size).init::<B>(&device);
        let mel = burn::tensor::Tensor::<B, 4>::zeros([1, 1, 128, 100], &device);
        let tokens = burn::tensor::Tensor::<B, 2, Int>::zeros([1, 16], &device);
        let prog = burn::tensor::Tensor::<B, 1, Int>::zeros([1], &device);
        let out = model.forward(mel, tokens, prog);
        let [b, s, v] = out.token_logits.dims();
        assert_eq!([b, s, v], [1, 16, vocab_size]);
        let _ = MusicFmEncoderConfig::smoke();
    }

    #[test]
    fn job_a_from_latents_shapes() {
        let device = Default::default();
        let d = 32usize;
        let k = 8usize;
        let model = DenoteJobAConfig::smoke(d, k).init::<B>(&device);
        let z = burn::tensor::Tensor::<B, 3>::zeros([2, 12, d], &device);
        let multi = burn::tensor::Tensor::<B, 2>::zeros([2, k], &device);
        let out = model.forward_from_latents(z, multi);
        assert_eq!(out.presence_logits.dims(), [2, k]);
    }

    #[test]
    fn miros_forward_from_latents_shapes() {
        let device = Default::default();
        let vocab_size = 4 + 205 + 128 + 128 + 128 + 8;
        let model = DenoteMirosConfig::smoke(vocab_size).init::<B>(&device);
        let d = model.encoder.d_model();
        let z = burn::tensor::Tensor::<B, 3>::zeros([1, 8, d], &device);
        let tokens = burn::tensor::Tensor::<B, 2, Int>::zeros([1, 16], &device);
        let prog = burn::tensor::Tensor::<B, 1, Int>::zeros([1], &device);
        let out = model.forward_from_latents(z, tokens, prog);
        assert_eq!(out.token_logits.dims()[2], vocab_size);
    }
}
