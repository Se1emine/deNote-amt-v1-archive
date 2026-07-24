//! Labels, piano-roll targets, token samples, and on-disk feature caches.
//!
//! Primary path for Denote-MIROS: [`token_cache`] (MIDI-event tokens).
//! Phase 1: [`latent_cache`] + [`manifest`] for Job L0 / Job A.
//! Phase 3: [`fluidr3`], [`stem_mix`], [`chunk`], [`curriculum`], [`programs`].
//! Legacy frame-level CRNN path: [`cache`] + [`labels`] (kept for ablations).

mod cache;
mod chunk;
mod curriculum;
mod fluidr3;
mod group_tokens;
mod labels;
mod latent_cache;
mod manifest;
mod note;
mod piano_roll;
mod programs;
mod stem_mix;
mod token_cache;

pub use cache::{save_notes_json, CachedSample, FeatureCache, SampleMeta};
pub use chunk::{chunk_audio_notes, slice_notes, AudioChunk, ChunkConfig};
pub use curriculum::{
    filter_curriculum, mean_n_instruments, split_for_piece, CurriculumSchedule,
};
pub use fluidr3::{
    default_fluidsynth_bin, default_sf2, render_midi_batch, render_midi_fluidr3, FluidRenderOpts,
};
pub use group_tokens::{
    build_group_token_sample, build_group_tokens_for_manifest, encode_group_tokens,
    filter_notes_by_programs, group_has_notes, group_token_rel_path, load_notes_for_entry,
    GroupTokenCache, GroupTokensBuildReport,
};
pub use labels::{notes_to_targets, FrameTargets, LabelConfig};
pub use latent_cache::{LatentFeatureCache, LatentMeta, LatentSample};
pub use manifest::{
    challenge_k, programs_multi_hot, split_for_id, unique_programs, CacheManifest, ManifestEntry,
    CHALLENGE_PROGRAM_NAMES,
};
pub use note::{NoteAnnotation, NoteList};
pub use piano_roll::{load_midi_notes, notes_from_json};
pub use programs::{
    challenge_to_gm, gm_to_challenge, is_drum_program, remap_notes_to_challenge, CHALLENGE_K,
};
pub use stem_mix::{
    discover_babyslakh, generate_stem_mix_chunks, mix_stems, write_chunks_flat, BabySlakhTrack,
    StemInfo, StemMixConfig,
};
pub use token_cache::{
    note_annotation_to_token_note, save_tokens_json, token_note_to_annotation,
    vocab_config_from_meta, TokenCachedSample, TokenFeatureCache, TokenSampleMeta,
};
