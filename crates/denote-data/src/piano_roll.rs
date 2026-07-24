use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

use crate::note::{NoteAnnotation, NoteList};

/// Parse a Standard MIDI File into note annotations (seconds).
///
/// Tracks `ProgramChange` per channel so each note gets a **GM program** id.
/// Channel 9 (0-based 9) is marked program 128 (drums) and typically filtered later.
///
/// Uses a simple tempo map (defaults to 120 BPM if no SetTempo).
pub fn load_midi_notes(path: impl AsRef<Path>) -> Result<NoteList> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read midi {}", path.display()))?;
    let smf = Smf::parse(&bytes).map_err(|e| anyhow!("midi parse: {e}"))?;

    let ticks_per_beat = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as f64,
        Timing::Timecode(fps, subframe) => {
            let _ = (fps, subframe);
            480.0
        }
    };

    let mut tempos: Vec<(u32, u32)> = vec![(0, 500_000)];
    for track in &smf.tracks {
        let mut abs_tick = 0u32;
        for ev in track {
            abs_tick += ev.delta.as_int();
            if let TrackEventKind::Meta(MetaMessage::Tempo(t)) = ev.kind {
                tempos.push((abs_tick, t.as_int()));
            }
        }
    }
    tempos.sort_by_key(|(t, _)| *t);
    tempos.dedup_by_key(|(t, _)| *t);

    let tick_to_sec = |tick: u32| -> f64 {
        let mut sec = 0.0f64;
        let mut prev_tick = 0u32;
        let mut us_pq = 500_000u32;
        for &(tt, tempo) in &tempos {
            if tt >= tick {
                break;
            }
            let dt = tt.saturating_sub(prev_tick) as f64;
            sec += dt * (us_pq as f64) / 1_000_000.0 / ticks_per_beat;
            prev_tick = tt;
            us_pq = tempo;
        }
        let dt = tick.saturating_sub(prev_tick) as f64;
        sec += dt * (us_pq as f64) / 1_000_000.0 / ticks_per_beat;
        sec
    };

    // Active notes: (channel, key) -> (start_tick, velocity, program)
    let mut active: std::collections::HashMap<(u8, u8), (u32, u8, u8)> =
        std::collections::HashMap::new();
    let mut notes = Vec::new();
    // Per-channel GM program (default 0)
    let mut channel_program = [0u8; 16];

    for track in &smf.tracks {
        let mut abs_tick = 0u32;
        active.clear();
        for ev in track {
            abs_tick += ev.delta.as_int();
            match ev.kind {
                TrackEventKind::Midi { channel, message } => {
                    let ch = channel.as_int();
                    match message {
                        MidiMessage::ProgramChange { program } => {
                            if (ch as usize) < 16 {
                                channel_program[ch as usize] = program.as_int();
                            }
                        }
                        MidiMessage::NoteOn { key, vel } => {
                            let k = key.as_int();
                            let v = vel.as_int();
                            let prog = if ch == 9 {
                                128 // drums
                            } else {
                                channel_program[ch as usize]
                            };
                            if v == 0 {
                                if let Some((st, vel0, p)) = active.remove(&(ch, k)) {
                                    notes.push(NoteAnnotation {
                                        start_sec: tick_to_sec(st),
                                        end_sec: tick_to_sec(abs_tick),
                                        pitch_midi: k,
                                        velocity: vel0,
                                        confidence: 1.0,
                                        program: p,
                                    });
                                }
                            } else if let Some((st, vel0, p)) =
                                active.insert((ch, k), (abs_tick, v, prog))
                            {
                                notes.push(NoteAnnotation {
                                    start_sec: tick_to_sec(st),
                                    end_sec: tick_to_sec(abs_tick),
                                    pitch_midi: k,
                                    velocity: vel0,
                                    confidence: 1.0,
                                    program: p,
                                });
                            }
                        }
                        MidiMessage::NoteOff { key, .. } => {
                            let k = key.as_int();
                            if let Some((st, vel0, p)) = active.remove(&(ch, k)) {
                                notes.push(NoteAnnotation {
                                    start_sec: tick_to_sec(st),
                                    end_sec: tick_to_sec(abs_tick),
                                    pitch_midi: k,
                                    velocity: vel0,
                                    confidence: 1.0,
                                    program: p,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        for ((_, k), (st, vel0, p)) in active.drain() {
            notes.push(NoteAnnotation {
                start_sec: tick_to_sec(st),
                end_sec: tick_to_sec(abs_tick),
                pitch_midi: k,
                velocity: vel0,
                confidence: 1.0,
                program: p,
            });
        }
    }

    notes.retain(|n| n.end_sec > n.start_sec);
    notes.sort_by(|a, b| {
        a.start_sec
            .partial_cmp(&b.start_sec)
            .unwrap()
            .then_with(|| a.pitch_midi.cmp(&b.pitch_midi))
            .then_with(|| a.program.cmp(&b.program))
    });

    Ok(NoteList {
        track_id: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("track")
            .to_string(),
        sr: 0,
        notes,
    })
}

/// Load notes from JSON (NoteList schema).
pub fn notes_from_json(path: impl AsRef<Path>) -> Result<NoteList> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let list: NoteList = serde_json::from_str(&text).context("parse note json")?;
    Ok(list)
}
