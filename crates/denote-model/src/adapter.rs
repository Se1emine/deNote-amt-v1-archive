//! Instrument-conditioned adapter over encoder latents (MIROS-style + aux heads).

use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        Lstm, LstmConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct AdapterConfig {
    pub d_model: usize,
    pub n_programs: usize,
    #[config(default = 128)]
    pub n_pitches_aux: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Adapts frozen/pretrained encoder states with program conditioning.
///
/// Improvements over plain MIROS adapter:
/// - optional stem/program embedding broadcast
/// - auxiliary frame pitch + onset heads (stabilize pitch before token CE)
#[derive(Module, Debug)]
pub struct InstrumentAdapter<B: Backend> {
    program_emb: Embedding<B>,
    proj_in: Linear<B>,
    rnn: Lstm<B>,
    proj_out: Linear<B>,
    norm: LayerNorm<B>,
    aux_pitch: Linear<B>,
    aux_onset: Linear<B>,
    act: Gelu,
    #[module(skip)]
    pub d_model: usize,
}

/// Adapter outputs used by decoder training / aux multi-task.
pub struct AdapterOutput<B: Backend> {
    /// Cross-attention memory `[B, T, D]`.
    pub memory: Tensor<B, 3>,
    /// Aux pitch logits `[B, T, n_pitches]`.
    pub pitch_logits: Tensor<B, 3>,
    /// Aux onset logits `[B, T, n_pitches]`.
    pub onset_logits: Tensor<B, 3>,
}

impl AdapterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InstrumentAdapter<B> {
        InstrumentAdapter {
            program_emb: EmbeddingConfig::new(self.n_programs.max(1), self.d_model).init(device),
            proj_in: LinearConfig::new(self.d_model, self.d_model).init(device),
            // unidirectional LSTM as recurrent adapter (hidden = d_model)
            rnn: LstmConfig::new(self.d_model, self.d_model, true).init(device),
            proj_out: LinearConfig::new(self.d_model, self.d_model).init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            aux_pitch: LinearConfig::new(self.d_model, self.n_pitches_aux).init(device),
            aux_onset: LinearConfig::new(self.d_model, self.n_pitches_aux).init(device),
            act: Gelu::new(),
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> InstrumentAdapter<B> {
    /// # Shapes
    /// - `latents`: `[B, T, D]` encoder output
    /// - `program_ids`: `[B]` instrument group ids
    pub fn forward(&self, latents: Tensor<B, 3>, program_ids: Tensor<B, 1, Int>) -> AdapterOutput<B> {
        let [b, t, _d] = latents.dims();
        let prog = self.program_emb.forward(program_ids.reshape([b, 1])); // [B,1,D]
        let prog = prog.repeat_dim(1, t); // [B,T,D]
        self.forward_with_prog(latents, prog)
    }

    /// Multi-instrument conditioning: **sum of active program embeddings** (Phase 1 default).
    ///
    /// - `latents`: `[B, T, D]`
    /// - `multi_hot`: `[B, K]` floats (0/1); K must match embedding table size
    pub fn forward_multi_hot(
        &self,
        latents: Tensor<B, 3>,
        multi_hot: Tensor<B, 2>,
    ) -> AdapterOutput<B> {
        let [b, t, d] = latents.dims();
        let [b2, k] = multi_hot.dims();
        assert_eq!(b, b2, "batch mismatch latents vs multi_hot");
        // Lookup all program embeddings then weighted sum: multi_hot @ E
        // Embedding expects [batch, seq] → [batch, seq, d]
        let ids: Vec<i32> = (0..k as i32).collect();
        let device = latents.device();
        let id_t = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, k]);
        let all_emb = self.program_emb.forward(id_t); // [1, K, D]
        let all_emb = all_emb.reshape([k, d]);
        // [B, K] @ [K, D] → [B, D]
        let prog = multi_hot.matmul(all_emb); // [B, D]
        let prog = prog.reshape([b, 1, d]).repeat_dim(1, t); // [B, T, D]
        self.forward_with_prog(latents, prog)
    }

    fn forward_with_prog(&self, latents: Tensor<B, 3>, prog: Tensor<B, 3>) -> AdapterOutput<B> {
        let x = latents + prog;
        let x = self.act.forward(self.proj_in.forward(x));
        let (h, _) = self.rnn.forward(x, None);
        let memory = self.norm.forward(self.proj_out.forward(h));
        let pitch_logits = self.aux_pitch.forward(memory.clone());
        let onset_logits = self.aux_onset.forward(memory.clone());
        AdapterOutput {
            memory,
            pitch_logits,
            onset_logits,
        }
    }
}
