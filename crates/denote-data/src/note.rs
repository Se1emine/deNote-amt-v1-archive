use serde::{Deserialize, Serialize};

/// A single ground-truth or predicted note event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoteAnnotation {
    pub start_sec: f64,
    pub end_sec: f64,
    pub pitch_midi: u8,
    #[serde(default)]
    pub velocity: u8,
    #[serde(default = "default_conf")]
    pub confidence: f32,
    /// Challenge program index (0..K-1). Default 0 for legacy JSON without field.
    #[serde(default)]
    pub program: u8,
}

fn default_conf() -> f32 {
    1.0
}

impl NoteAnnotation {
    pub fn duration_sec(&self) -> f64 {
        (self.end_sec - self.start_sec).max(0.0)
    }
}

/// Collection of notes for one track.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoteList {
    pub track_id: String,
    pub sr: u32,
    pub notes: Vec<NoteAnnotation>,
}
