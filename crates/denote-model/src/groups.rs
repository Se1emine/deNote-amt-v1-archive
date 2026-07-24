//! Instrument groups for modular decoders (Phase 2).
//!
//! Config-driven program → group mapping (master §2.6 / challenge K=8).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

/// One instrument group `g` with a set of challenge program indices.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GroupDef {
    pub id: u32,
    pub name: String,
    pub programs: Vec<u8>,
}

/// Full groups table + reverse map program → group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupsConfig {
    pub groups: Vec<GroupDef>,
    /// program id → group id
    program_to_group: HashMap<u8, u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct GroupsToml {
    #[serde(rename = "group", default)]
    groups: Vec<GroupDef>,
}

impl GroupsConfig {
    pub fn new(groups: Vec<GroupDef>) -> Self {
        let mut program_to_group = HashMap::new();
        for g in &groups {
            for &p in &g.programs {
                program_to_group.insert(p, g.id);
            }
        }
        Self {
            groups,
            program_to_group,
        }
    }

    /// Challenge default G=4 (keys / strings / wood / brass).
    pub fn challenge8() -> Self {
        Self::new(vec![
            GroupDef {
                id: 0,
                name: "keys".into(),
                programs: vec![0],
            },
            GroupDef {
                id: 1,
                name: "strings".into(),
                programs: vec![1, 2, 3],
            },
            GroupDef {
                id: 2,
                name: "wood".into(),
                programs: vec![4, 5, 7],
            },
            GroupDef {
                id: 3,
                name: "brass".into(),
                programs: vec![6],
            },
        ])
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read groups {}: {e}", path.display()))?;
        Self::parse_toml(&text)
    }

    pub fn parse_toml(text: &str) -> Result<Self, String> {
        let raw: GroupsToml =
            toml::from_str(text).map_err(|e| format!("parse groups toml: {e}"))?;
        if raw.groups.is_empty() {
            return Err("groups config has no [[group]] entries".into());
        }
        Ok(Self::new(raw.groups))
    }

    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    pub fn group(&self, id: u32) -> Option<&GroupDef> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Map program index → group id, if known.
    pub fn program_to_group(&self, program: u8) -> Option<u32> {
        self.program_to_group.get(&program).copied()
    }

    /// Programs belonging to group `g`.
    pub fn programs_of(&self, group_id: u32) -> &[u8] {
        self.group(group_id)
            .map(|g| g.programs.as_slice())
            .unwrap_or(&[])
    }

    /// Whether any of `programs` fall into group `g`.
    pub fn group_has_any(&self, group_id: u32, programs: &[u8]) -> bool {
        let set = self.programs_of(group_id);
        programs.iter().any(|p| set.contains(p))
    }

    /// Multi-hot mask of programs in group that are active in the clip.
    /// Length = `k` (presence dimension); non-group programs stay 0.
    pub fn group_active_multi_hot(&self, group_id: u32, active_programs: &[u8], k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; k];
        let in_g = self.programs_of(group_id);
        for &p in active_programs {
            let i = p as usize;
            if i < k && in_g.contains(&p) {
                y[i] = 1.0;
            }
        }
        y
    }

    /// First / primary program of the group (for FSM bookkeeping).
    pub fn primary_program(&self, group_id: u32) -> Option<u8> {
        self.programs_of(group_id).first().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge8_map() {
        let g = GroupsConfig::challenge8();
        assert_eq!(g.n_groups(), 4);
        assert_eq!(g.program_to_group(0), Some(0));
        assert_eq!(g.program_to_group(1), Some(1));
        assert_eq!(g.program_to_group(2), Some(1));
        assert_eq!(g.program_to_group(3), Some(1));
        assert_eq!(g.program_to_group(4), Some(2));
        assert_eq!(g.program_to_group(5), Some(2));
        assert_eq!(g.program_to_group(7), Some(2));
        assert_eq!(g.program_to_group(6), Some(3));
        assert_eq!(g.program_to_group(99), None);
    }

    #[test]
    fn parse_toml_roundtrip() {
        let text = r#"
[[group]]
id = 0
name = "keys"
programs = [0]

[[group]]
id = 1
name = "strings"
programs = [1, 2, 3]
"#;
        let g = GroupsConfig::parse_toml(text).unwrap();
        assert_eq!(g.n_groups(), 2);
        assert_eq!(g.program_to_group(2), Some(1));
    }

    #[test]
    fn group_active_multi_hot() {
        let g = GroupsConfig::challenge8();
        let y = g.group_active_multi_hot(1, &[0, 1, 4], 8);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[1], 1.0);
        assert_eq!(y[4], 0.0);
    }
}
