//! T5-style token decoder with cross-attention over adapter memory.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput},
    },
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct TokenDecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Autoregressive MIDI-token decoder.
#[derive(Module, Debug)]
pub struct TokenDecoder<B: Backend> {
    token_emb: Embedding<B>,
    decoder: TransformerDecoder<B>,
    lm_head: Linear<B>,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub d_model: usize,
}

impl TokenDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenDecoder<B> {
        TokenDecoder {
            token_emb: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            decoder: TransformerDecoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_dropout(self.dropout)
            .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            vocab_size: self.vocab_size,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> TokenDecoder<B> {
    /// Teacher-forcing forward.
    ///
    /// # Shapes
    /// - `tokens`: `[B, S]` input token ids (usually targets shifted right, starting with BOS)
    /// - `memory`: `[B, T, D]` encoder/adapter memory
    /// - returns logits: `[B, S, V]`
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, memory: Tensor<B, 3>) -> Tensor<B, 3> {
        let emb = self.token_emb.forward(tokens);
        let h = self
            .decoder
            .forward(TransformerDecoderInput::new(emb, memory));
        self.lm_head.forward(h)
    }
}
