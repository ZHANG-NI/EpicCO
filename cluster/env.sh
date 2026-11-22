# Site configuration for running the framework on the Slurm cluster. Source it from every job script:  . cluster/env.sh
# Edit the block marked EDIT if the cluster differs from what is assumed here.

# ── EDIT ────────────────────────────────────────────────────────────────────────────────────────────────────────────
export SE_ROOT="${SE_ROOT:-$HOME/alphaco}"          # where the bundles were unpacked
# ~/alphaco may be a symbolic link into the scratch file system (the home quota, shared with other
# projects, filled up and every job of the account died at its next write). Every process uses the PHYSICAL path.
SE_ROOT="$(cd "$SE_ROOT" 2>/dev/null && pwd -P || echo "$SE_ROOT")"; export SE_ROOT
SE_MODULES="${SE_MODULES-}"                               # compiler module(s) to load; empty = the system g++ (here: /usr/bin/g++ 11.5)
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

if [ -n "$SE_MODULES" ] && type module >/dev/null 2>&1; then
  for m in $SE_MODULES; do module load "$m" 2>/dev/null || echo "env.sh: module $m could not be loaded" >&2; done
fi
GXX=$(command -v g++)
export SE_TOOLCHAIN_BIN="$(dirname "${GXX:-/usr/bin/g++}")"   # judge / final evaluation: PATH prefix for the official build
export SE_TOOLCHAIN=""                                        # no separate toolchain directory to bind into the jail
# Rust host (2D guillotine packing): the toolchain installed under the project root, with the vendored crates in
# <toolchain>/vendor_gdrr; bound into the jail as its toolchain directory and put first on the judge's build PATH.
SE_RUST_TC="$SE_ROOT/common/rust/rustup/toolchains/stable-x86_64-unknown-linux-gnu"
if [ -x "$SE_RUST_TC/bin/cargo" ]; then
  export SE_TOOLCHAIN="$SE_RUST_TC"
  export SE_TOOLCHAIN_BIN="$SE_RUST_TC/bin:$SE_TOOLCHAIN_BIN"
fi
export SE_JAIL_PATH="$PATH"                                   # the jail sees the same compiler as the judge
export SE_JAIL_LD="${LD_LIBRARY_PATH:-}"
export SE_REAL_HOME="/home/agent"                             # the agent's home directory inside the jail
# hidden inside the jail: every user's home (the project root with the benchmark and the judge's files lives there)
export SE_HIDE="/home:/data:/scratch:$(dirname "$HOME")"
# the OAuth credentials the sessions use: a copy inside the project root (not in the home directory: a write to a full
# home quota truncates the file)
[ -s "$SE_ROOT/common/cred/.credentials.json" ] && export SE_CRED="$SE_ROOT/common/cred/.credentials.json"
export PYTHONUNBUFFERED=1
