#!/usr/bin/env bash
# Verifies that ruvector-wasm's `lattice-simd` feature actually vectorizes on
# wasm32: builds the crate with and without `-C target-feature=+simd128` and
# counts SIMD128 opcodes in each emitted .wasm artifact. Fails closed on any
# missing prerequisite, missing/empty artifact, or unmet opcode condition.
set -euo pipefail

PACKAGE="ruvector-wasm"
RUST_TARGET="wasm32-unknown-unknown"
FEATURE="lattice-simd"
ARTIFACT="ruvector_wasm.wasm"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_TARGET_DIR="${CARGO_TARGET_DIR:-target}/wasm-simd-check"
SIMD_TARGET_DIR="$BASE_TARGET_DIR/simd128"
CONTROL_TARGET_DIR="$BASE_TARGET_DIR/control"

echo "== Prerequisites =="

if ! rustup target list --installed 2>/dev/null | grep -qx "$RUST_TARGET"; then
  echo "FAIL: rust target '$RUST_TARGET' is not installed." >&2
  echo "      Install it with: rustup target add $RUST_TARGET" >&2
  exit 1
fi
echo "OK: $RUST_TARGET target installed"

if ! command -v wasm-objdump >/dev/null 2>&1; then
  echo "FAIL: wasm-objdump not found on PATH (ships with WABT)." >&2
  echo "      Install it with: brew install wabt (macOS) or apt-get install wabt (Linux)." >&2
  exit 1
fi
echo "OK: wasm-objdump found at $(command -v wasm-objdump)"

count_simd128_opcodes() {
  local wasm_file="$1"
  if [[ ! -s "$wasm_file" ]]; then
    echo "FAIL: expected wasm artifact at '$wasm_file' but it is missing or empty." >&2
    exit 1
  fi

  local disasm
  if ! disasm="$(wasm-objdump -d "$wasm_file")"; then
    echo "FAIL: wasm-objdump could not disassemble '$wasm_file'." >&2
    exit 1
  fi

  # grep -c exits 1 on zero matches, which is a legitimate count here (the
  # control arm below), so neutralize that with `|| true` rather than
  # letting set -e/pipefail treat "no opcodes found" as a script error.
  local count
  count="$(grep -Ec '\b(v128|i8x16|i16x8|i32x4|i64x2|f32x4|f64x2)\.[a-z_0-9]+' <<<"$disasm" || true)"
  echo "$count"
}

build_artifact() {
  local target_dir="$1"
  local rustflags="$2"
  CARGO_TARGET_DIR="$target_dir" RUSTFLAGS="$rustflags" \
    cargo build --release -p "$PACKAGE" --target "$RUST_TARGET" --features "$FEATURE" 1>&2
  echo "$target_dir/$RUST_TARGET/release/$ARTIFACT"
}

echo ""
echo "== Arm A: RUSTFLAGS='-C target-feature=+simd128', --features $FEATURE =="
SIMD_WASM="$(build_artifact "$SIMD_TARGET_DIR" "-C target-feature=+simd128")"
SIMD_COUNT="$(count_simd128_opcodes "$SIMD_WASM")"
echo "SIMD128 opcode count: $SIMD_COUNT"
if (( SIMD_COUNT <= 0 )); then
  echo "FAIL: expected > 0 SIMD128 opcodes with +simd128 and --features $FEATURE, got $SIMD_COUNT." >&2
  exit 1
fi

echo ""
echo "== Arm B (control): no target-feature flag, --features $FEATURE =="
CONTROL_WASM="$(build_artifact "$CONTROL_TARGET_DIR" "")"
CONTROL_COUNT="$(count_simd128_opcodes "$CONTROL_WASM")"
echo "SIMD128 opcode count: $CONTROL_COUNT"

# Without the target-feature flag, lattice-embed's wasm32 kernels take their
# scalar fallback (crates/ruvector-core/src/distance.rs), so this arm
# currently measures 0 SIMD128 opcodes. A future dependency could
# legitimately contribute some vector code even without the flag, so this
# asserts the delta direction (control strictly below the +simd128 build)
# rather than hard-coding zero.
if (( CONTROL_COUNT >= SIMD_COUNT )); then
  echo "FAIL: expected the control build to carry fewer SIMD128 opcodes than the +simd128 build (control=$CONTROL_COUNT, simd128=$SIMD_COUNT)." >&2
  exit 1
fi

echo ""
echo "== PASS =="
echo "+simd128 build: $SIMD_COUNT SIMD128 opcodes"
echo "control build:  $CONTROL_COUNT SIMD128 opcodes"
