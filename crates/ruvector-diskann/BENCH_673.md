# #673 — PQ-guided search benchmark log

Evidence trail for wiring `pq_asymmetric_distance` into `DiskAnnIndex::search()`'s
traversal (see `graph.rs::greedy_search_core` / `greedy_search_pq_fast` and
`index.rs::search()`). Two rounds: round 1 measured the naive PQ-guided arm
against the exact-traversal baseline; round 2 is a scoped query-time tuning
sweep gated on a pre-registered threshold, run only because round 1's gap
(0.816 -> 0.648 recall@10) is consistent with an under-provisioned candidate
pool rather than a broken idea — classic DiskANN literature holds recall with
PQ-steered traversal + exact re-rank when the search list is wide enough.

All numbers: Apple M2 Max, release build (`cargo test --release`), same
dataset/query seeds across both rounds unless noted.

## Round 1 — naive PQ-guided vs exact (measured)

N=100,000, dim=128, M=16 PQ subspaces, seeds `0xD15CA77` (dataset), `0xBEEF`
(ground-truth query sample, 50 queries), `0xFACEB00C` (latency query sample,
20 warmup + 200 timed). Harness: `bench_pq_vs_exact_100k` in `src/index.rs`.

| Arm | Build | Recall@10 | Median | p95 |
|---|---|---|---|---|
| Exact traversal (pq_subspaces=0) | 840.7s | 0.816 | 695us | 2010us |
| PQ-guided (pq_subspaces=16, search_beam=64, full-beam rerank) | 1264.1s | 0.648 | 274us | 2296us |

Verdict: does not clear the original pre-registered gate (recall@10 >= 0.85).
PR not opened; mutation test confirmed the wiring is correct (corrupting the
distance table collapses recall 0.900 -> 0.030 on the smaller 2k/64d harness),
so the gap is a config/approximation-quality issue, not a bug.

## Round 2 — query-time tuning sweep

### Pre-registration (2026-07-12T17:48:21Z, before any sweep code was run)

**Tuning gate** (all three must hold simultaneously, at N=100k/dim=128, same
seeds as round 1): a config is a PASS iff

- `recall@10 >= 0.80` (registered against round 1's *measured* exact-arm
  ceiling of 0.816, i.e. within 0.02 of it — not an absolute recall bar)
- `median latency <= 0.6 * 695us = 417us` (round-1 exact-arm median)
- `p95 latency <= 1.15 * 2010us = 2311.5us` (round-1 exact-arm p95)

If no config clears: final verdict is NO-OPEN, no further tuning rounds.
If a config clears: rebase-check vs ruvnet/RuVector#683, then open the PR
with round 1 + round 2 as combined A/B evidence.

**Sweep plan** (query-time only, no index rebuild against the round-1 M=16
build's config — one fresh N=100k/M=16 build for this round since round 1's
in-memory index was not persisted, then all cells below reuse that single
build):

- Search list width `L` in `{64 (current default), 128 (2x), 256 (4x)}` —
  the `beam_width` passed to `greedy_search_pq_fast`, which bounds both the
  PQ-guided traversal frontier and the candidate pool it returns.
- Exact re-rank pool size `R` in `{10 (=k), 30 (=3k), 100 (=10k)}` — take the
  `R` PQ-closest candidates (already sorted ascending by PQ distance) out of
  the `L`-wide traversal result, capped at `min(R, L)`, before exact-L2
  reranking to top-10. This decouples "how wide the PQ-guided walk explores"
  from "how many of those results pay the exact-L2 cost."
- 3x3 = 9 cells total, one shared M=16 build, one shared ground-truth set
  (same 50 queries, seed `0xBEEF`) and one shared latency query pool (same
  220 queries, seed `0xFACEB00C`) reused across all 9 cells.
- Escalation: only if the best cell's recall@10 is within ~0.05 of the gate
  (i.e. >= 0.75) but doesn't clear it, add exactly ONE additional rebuild arm
  at M=32 subspaces, re-measured at the best (L, R) cell found in the M=16
  sweep. No further sweeping at M=32.

### Results

Run: `cargo test -p ruvector-diskann --release -- --ignored --nocapture
bench_pq_tuning_sweep_100k` (harness in `src/index.rs`), 2026-07-12, Apple M2
Max, machine-wide exclusive build/test lane for the M=16 build and all nine
M=16 cells. One fresh M=16 build: **3734.2s**; all nine cells reuse it, plus
the shared ground-truth (50 queries, seed `0xBEEF`) and latency pool (220
queries, seed `0xFACEB00C`) per the pre-registration.

| L | R | Recall@10 | Median | p95 | Gate |
|---|---|---|---|---|---|
| 64 | 10 | 0.322 | 155us | 462us | FAIL (recall) |
| 64 | 30 | 0.526 | 154us | 458us | FAIL (recall) |
| 64 | 100 | 0.676 | 166us | 1102us | FAIL (recall) |
| 128 | 10 | 0.330 | 274us | 1962us | FAIL (recall) |
| 128 | 30 | 0.520 | 309us | 2083us | FAIL (recall) |
| 128 | 100 | 0.740 | 265us | 1353us | FAIL (recall) |
| 256 | 10 | 0.330 | 512us | 1752us | FAIL (recall, median) |
| 256 | 30 | 0.512 | 386us | 1022us | FAIL (recall) |
| 256 | 100 | **0.756** | 488us | 2271us | FAIL (recall, median) |

Best-by-recall cell: L=256 R=100 at 0.756 — within 0.05 of the 0.80 gate, so
the pre-registered single M=32 escalation arm fired at that cell:

| Arm | Build | Recall@10 | Median | p95 | Gate |
|---|---|---|---|---|---|
| M=32, L=256, R=100 | 2843.8s | **0.922** | 680us | 2676us | FAIL (median, p95) |

Confound disclosure (M=32 arm only): the M=32 timed pass ran in a window with
other build/upload jobs active on the machine after a mid-run widening of the
build-lane policy; the M=16 cells ran quieter. This cannot rescue the verdict:
the M=16 best cell already missed the 417us median gate at 488us under the
quieter window, and M=32 strictly increases per-hop distance-table work (2x
subquantizers), so no plausible decontended M=32 median lands under 417us.
Recall is deterministic and unaffected by contention.

### Verdict

**No cell clears the pre-registered gate (all three thresholds
simultaneously). Final verdict per pre-registration: NO-OPEN.** No further
tuning rounds.

Two findings worth recording:

1. **The latency failure is structural at dim=128.** PQ-guided traversal saves
   exact-L2 work per hop but pays a distance-table lookup per neighbor; at
   d=128 with M=16 the net win only materializes at recall-destroying beam
   widths, and M=32 (needed for recall) doubles the table cost back to
   exact-arm latency (680us vs 695us exact median).
2. **M=32 PQ-guided + exact re-rank EXCEEDS the exact-traversal recall ceiling
   at parity latency**: 0.922 vs 0.816 recall@10, 680us vs 695us median. The
   PQ-steered frontier explores differently than the exact greedy walk and,
   with a generous candidate pool (L=256) plus exact re-rank, surfaces more
   true neighbors. As a *speed* feature (the #673 framing and this gate) the
   wiring does not pay; as a *recall* feature at equal cost it beats the
   baseline by +10.6pp. That repositioning — plus OPQ rotation, SIMD table
   scans, or PQ admission only for frontier expansion — is what a revisit
   would take.
