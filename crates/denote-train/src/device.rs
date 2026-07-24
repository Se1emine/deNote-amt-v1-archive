//! Backend / device selection for train CLI.

use anyhow::{bail, Result};
use log::info;

/// Resolved training device kind (compile-time features gate GPU backends).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Wgpu,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
            DeviceKind::Wgpu => "wgpu",
        }
    }
}

/// Parse `--device auto|cpu|cuda|wgpu` and check feature availability.
pub fn resolve_device(arg: &str) -> Result<DeviceKind> {
    let kind = match arg.trim().to_ascii_lowercase().as_str() {
        "auto" => resolve_auto(),
        "cpu" => DeviceKind::Cpu,
        "cuda" => DeviceKind::Cuda,
        "wgpu" => DeviceKind::Wgpu,
        other => bail!("unknown --device '{other}' (expected auto|cpu|cuda|wgpu)"),
    };
    ensure_feature(kind)?;
    Ok(kind)
}

fn resolve_auto() -> DeviceKind {
    #[cfg(feature = "cuda")]
    {
        return DeviceKind::Cuda;
    }
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        return DeviceKind::Wgpu;
    }
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        DeviceKind::Cpu
    }
}

fn ensure_feature(kind: DeviceKind) -> Result<()> {
    match kind {
        DeviceKind::Cpu => Ok(()),
        DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "device=cuda requested but binary built without feature `cuda`. \
                     Rebuild: cargo run -p denote-train --release --features cuda -- ..."
                );
            }
        }
        DeviceKind::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(())
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!(
                    "device=wgpu requested but binary built without feature `wgpu`. \
                     Rebuild: cargo run -p denote-train --release --features wgpu -- ..."
                );
            }
        }
    }
}

/// Log resolved backend for operators / diploma logs.
pub fn log_backend(kind: DeviceKind, context: &str) {
    let feature_cuda = cfg!(feature = "cuda");
    let feature_wgpu = cfg!(feature = "wgpu");
    info!(
        "{context} | backend={} device={} | features: cuda={} wgpu={} ndarray=true",
        kind.as_str(),
        kind.as_str(),
        feature_cuda,
        feature_wgpu
    );
}
