# PseudoGRN SERGIO density sweep — results analysis

Completed 9-run sweep (3 tiers x 3 seeds), Kaggle, 2026-09-08, 339.0 min total
(see `density_experiment_log.txt`). Results downloaded to
`results/pseudogrn_results/PseudoGRN_results/SERGIO/tier<N>_seed<S>/`.
This file was written after the sweep to check it against the same confound
already found and fixed in this project's RiTINI/MTGRN sweeps — not part of
the original run.

## Results

Baseline (chance-level AUPRC for this dataset): 1155 true edges / (37 TFs x
399 possible targets) = **0.0782** (same 400-gene/37-TF SERGIO dataset every
model in this project uses; same baseline value RiTINI's own analysis
already established).

| runid | AUPRC | AUROC | ratio vs baseline | n_cells_pooled |
|---|---|---|---|---|
| tier1_seed0 | 0.3316 | 0.7088 | 4.24x | 3000 |
| tier1_seed1 | 0.3547 | 0.7204 | 4.53x | 3000 |
| tier1_seed2 | 0.3452 | 0.7145 | 4.41x | 3000 |
| tier2_seed0 | 0.3769 | 0.7391 | 4.82x | 3000 |
| tier2_seed1 | 0.3587 | 0.7325 | 4.59x | 3000 |
| tier2_seed2 | 0.3615 | 0.7295 | 4.62x | 3000 |
| tier3_seed0 | 0.3535 | 0.7244 | 4.52x | 3000 |
| tier3_seed1 | 0.3573 | 0.7259 | 4.57x | 3000 |
| tier3_seed2 | 0.3882 | 0.7439 | 4.96x | 3000 |

| tier | AUPRC mean+/-std | AUROC mean+/-std | ratio |
|---|---|---|---|
| 1 (15 reps, 40,500 natural pool) | 0.3438+/-0.0095 | 0.7146+/-0.0047 | 4.39x |
| 2 (5 reps, 13,500 natural pool)  | 0.3657+/-0.0080 | 0.7337+/-0.0040 | 4.67x |
| 3 (3 reps, 8,100 natural pool)   | 0.3663+/-0.0156 | 0.7314+/-0.0089 | 4.68x |

**PseudoGRN clearly learns real structure** -- 4.2x-5.0x baseline across
every one of the 9 seeds, tight within-tier std, no failures/crashes.

## CONFOUND FOUND (same bug class as MTGRN's first Phase 2 sweep, found
## independently by checking this project's OWN precedent after fixing it
## there)

`CHANGES_max_cells_report.txt` (2026-09-08) documents that
`sergio_prepare_data.py`'s `--max_cells` default (3000) was added to keep
PseudoGRN's Mixed-KSG MI estimator tractable at SERGIO's pooled cell
counts. **But 3000 is smaller than even Tier 3's full natural pool
(8,100)** -- confirmed directly from every one of the 9 `metrics.json`
files: `n_cells_pooled: 3000` for ALL nine runs, regardless of tier.

**This means every tier in this sweep trained on the identical 3,000
cells.** The density variable (proposal Section 5.5, RQ2 -- how does
performance degrade as available replicates shrink) never actually
varied downstream of this cap. This is architecturally the exact same
bug independently introduced (and then caught and fixed) in this
project's MTGRN Phase 2 harness: a uniform absolute `--max_cells` cap
set below the sparsest tier's natural pool size silently equalizes every
tier's actual training data.

**Consistent with this diagnosis**: the observed per-tier AUPRC pattern
is FLAT-TO-SLIGHTLY-INVERTED (Tier 1: 4.39x, Tier 2: 4.67x, Tier 3:
4.68x) -- if real replicate-density degradation were present, Tier 1
(most data) should score noticeably HIGHER than Tier 3 (least), not
statistically indistinguishable or slightly lower. This is the same
"tiers collapse to the same result" signature that flagged the MTGRN bug
before this one was even suspected.

**Valid finding from this run**: PseudoGRN learns real regulatory
structure from real SERGIO data (4.2x-5.0x baseline, solid AUROC~0.71-0.74).
**Not answerable from this run**: the actual density-degradation question
this sweep exists to measure -- confounded by the cap, exactly like
MTGRN's first attempt.

## Recommended fix (same pattern already applied to MTGRN)

Raise `--max_cells` above Tier 3's natural pool (8,100) so only Tier 1
(and possibly Tier 2) get compressed, preserving the tier ordering. Given
this project's MTGRN fix used 22,500 (chosen from a real Kaggle GPU
timing smoke test, not guessed), and PseudoGRN's compute profile is
COMPLETELY DIFFERENT (Mixed-KSG MI estimator, pure-Python per-point
loop, NOT a GPU transformer forward/backward pass) -- **PseudoGRN's own
safe cap needs its own timing check, not a borrowed number**. The
original `CHANGES_max_cells_report.txt` already measured ~2s/candidate
pair at ~40k-window scale (extrapolated to ~8+ hours for Tier 1 alone,
uncapped) -- so raising the cap much above 3000 risks reintroducing the
original tractability problem this cap was built to solve. This is a
real tension (fix the confound vs. keep runs tractable) that needs a
user decision, not a default to just copy from MTGRN.

## STATUS: confound documented, NOT fixed or re-run here (per this
## project's policy: analysis only in this environment, training on
## Kaggle). This is the same class of bug as MTGRN's first sweep, found by
## checking PseudoGRN's own already-completed results against that
## precedent -- worth flagging to the user before this gets folded into
## any cross-model writeup, since the current numbers can speak to "does
## PseudoGRN work" but not "how does PseudoGRN degrade with density."

---

# Fix (pseudogrn_fix_and_reanalyze_prompt.md) -- progress

Per explicit user instruction: no code executed in this environment: all
training/timing happens on Kaggle. Everything below is either (a) answered
by reading the frozen source directly (no execution needed), (b) a code
change (not run here), or (c) a script written for the user to run on
Kaggle and report back -- gated tasks are marked as such, not guessed.

## Task 2 -- GPU investigation (research only, no execution needed for this part)

**2.1 -- NN search mechanism, confirmed by reading `Pseudo-network/psedoScore.py`
directly**: `MI_Gao()` (the Mixed-KSG estimator `cal_mi()`/`cal_mi2()` calls
per candidate pair) uses `scipy.spatial.cKDTree`, THREE trees per pair
(`tree_xy`, `tree_x`, `tree_y`), with:
  - ONE vectorized `tree_xy.query(point, k+1, p=inf)` call per point (list
    comprehension over all N samples, not the flagged bottleneck).
  - A Python `for i in range(N)` loop (lines 46-56) that, for EVERY
    sample, does 1-2 `query_ball_point(..., p=inf)` calls -- THIS is the
    "per-point Python loop" the original report flagged. Distance metric
    throughout is **Chebyshev / L-infinity** (`p=float('inf')`), not the
    more common Euclidean/L2.

**2.2 -- does a GPU drop-in exist?** No clean one, for a specific reason:
mainstream GPU nearest-neighbor libraries (FAISS, RAPIDS cuML, PyTorch's
own ops) are built around L2/inner-product metrics for k-NN search;
**exact Chebyshev-distance ball/radius queries (`query_ball_point`, used
here for exact epsilon-neighbor COUNTS, not just top-k) are not a
first-class primitive in any of them.** A correct GPU port would mean
hand-writing a custom kernel (pairwise Chebyshev distance via
broadcasting + threshold-count, feasible in raw torch) -- not importing
an existing validated library. That shifts this from "swap a backend"
to "write and prove correct a new numerical implementation of the same
primitive," which is exactly the correctness risk Task 2's own
instructions flag as a reason to prefer NOT forcing it.

**2.3 -- decision: no GPU path pursued**, per the prompt's own explicit
escape valve ("if no clean, validated GPU path exists... say so plainly
and move to Task 3"). Not attempting a custom kernel + validation harness
under today's no-local-execution constraint would mean asking the user to
run and eyeball-verify numerically-sensitive correctness checks on
Kaggle with no way for me to iterate on failures locally -- too much risk
for today's deadline. Moving to Task 3.

## Task 3 -- CPU parallelization: ALREADY EXISTS, not new work

Confirmed by reading `psedoScore.py` directly: `cal_mi2()` and `MRMR2()`
BOTH already parallelize across the 14,763 independent candidate pairs
via `pqdm.processes.pqdm(params, ..., n_jobs=n_jobs)` -- this is the
frozen library's OWN mechanism, already wired through this harness's
`--n_jobs` flag (the original sweep already used `--n_jobs 4`, visible in
`density_experiment_log.txt`). Nothing to build here.

**One real fix applied**: `--n_jobs` defaulted to a hardcoded `1`
(train_sergio.py) / `1` (run_density_experiment.py) rather than the
"actual core count of the Kaggle instance" the prompt asks for. Changed
both to `default=os.cpu_count() or 1` -- auto-detects at runtime instead
of requiring the user to know and pass their instance's core count
manually. Does not change results, only wall-clock time. Verified
(`py_compile` + a plain import, no data touched) that both files still
import and parse cleanly after the change.

## Task 1 -- real timing curve: SCRIPT WRITTEN, NOT YET RUN (gated on Kaggle)

Wrote `benchmark_max_cells_timing.py` -- times `smooth()` + `cal_mi2()` +
`MRMR2()` (via `train_sergio.py`'s own imports, so the compat-patched
`smooth()` is exercised exactly as the real pipeline uses it) at pool
sizes 3000/6000/9000/13500/20000/27000/40500, subsampled from ONE
uncapped Tier-1 source dataset (DPT computed once, reused for every pool
size tested -- avoids paying DPT's cost repeatedly). Verified the script
imports cleanly and picks up `train_sergio`'s `PreprocessData.smooth`
monkeypatch correctly (checked directly, no data involved). NOT run --
needs real SERGIO data, which only exists on Kaggle.

## Tasks 4, 5, 6 -- GATED on Task 1's real numbers, not done yet

Per the prompt's own instructions, the cap (Task 4) must come FROM Task
1's real timing curve, the validation gate (Task 5) must show REAL
`n_cells_pooled` numbers from 3 actual test runs, and the full re-run
(Task 6) only happens after Task 5 passes. None of this can be responsibly
done without the numbers Task 1's script produces -- not attempting to
guess a cap the way the original 3000 default was guessed (from a small
synthetic stand-in, which is exactly how this bug happened in the first
place). Waiting on the user to run Task 1 on Kaggle and report back.

## STATUS: Tasks 1 (script ready), 2 (answered), 3 (confirmed + fixed)
## done. Tasks 4-6 blocked on real Task 1 timing data from Kaggle.

---

# Task 1 partial real data + user decision to stop investigating

Ran `benchmark_max_cells_timing.py`'s first point on real Kaggle data
(4 cores) before stopping: **pool_size=3000 -> smooth()=445.2s (7.4min),
cal_mi2()=1451.6s (24.2min), MRMR2()=47.0s, TOTAL=1943.8s (32.4min).**
Both smooth() and cal_mi2() came in considerably slower than the
original synthetic-extrapolated estimate (~5min/~10min guessed vs.
7.4min/24.2min real) -- confirms the original 3000 default's problem
wasn't just "too small a cap," the underlying per-stage cost model it was
based on was also an underestimate.

Linear-scaling projection from this single real point (smooth: 0.1486
s/window; cal_mi2: 0.4845 s/window aggregate at n_jobs=4; MRMR:
0.0157 s/window) -- NOT verified beyond one data point, could be worse:

| pool size | projected total | 
|---|---|
| 13,500 (Tier 2 natural) | ~2.43h |
| 22,000 (chosen Tier 1 cap) | ~3.96h |
| 40,500 (Tier 1 natural) | ~7.30h (matches the original stuck-run's own ~8h estimate) |

**User decision: stop further timing investigation and the GPU path here
(Task 2 already closed above; remaining Task 1 points and Task 4's own
data-driven cap selection both skipped).** Proceeding directly with fixed
caps instead:
- **Tier 1: capped to 22,000** (`--max_cells 22000`, now the default in
  both `sergio_prepare_data.py` and `run_density_experiment.py`).
- **Tier 2 and Tier 3: uncapped at natural size** (13,500 / 8,100) --
  automatic consequence of `min(natural, cap)` with cap=22,000, no
  special-casing needed (verified again numerically: 22000/13500/8100).
- **Single seed (0) per tier, 3 runs total** -- not the full 9-run sweep,
  per explicit user instruction, given the projected per-run cost above.

## Task 3 (parallelization) re-confirmed: nothing to add

Already fixed earlier in this file's Fix section (`--n_jobs` now
defaults to `os.cpu_count()`, already the frozen library's own pqdm-based
parallelization, not new code). User asked to add it "if trivial, skip if
it adds complexity" -- it was already done before this message, so
nothing further was needed.

## Task 5 validation gate: script written, not yet run (gated on Kaggle)

Wrote `check_cell_counts.py` -- reads `n_cells_pooled_full`/
`n_cells_written` back from each of the 3 runs' `prep_meta.json` and
`n_cells_pooled` from `metrics.json`, prints a direct side-by-side table,
and explicitly checks (not just assumes): (a) all 3 `n_cells_written`
values are distinct, (b) they're correctly ordered Tier1 > Tier2 > Tier3,
(c) `prep_meta.json` and `metrics.json` agree on the written count. This
is exactly the check that would have caught the original bug (every tier
silently capped to 3000) before a full sweep, not after.

Verified (`py_compile` + a plain arithmetic check of the min(natural,cap)
logic, no data touched): with cap=22000, the three tiers resolve to
22000/13500/8100 -- confirmed distinct and correctly ordered.

## STATUS: all code/doc changes done, nothing executed. 3 launch commands
## below -- run in order, report back the timings and check_cell_counts.py's
## output (or just its PASS/FAIL lines) before this gets marked verified.

## LAUNCH COMMANDS (Kaggle, in order)

```
# 1. Run the corrected 3-run sweep (single seed, --max_cells now defaults
#    to 22000 and --n_jobs to the detected core count -- shown explicitly
#    for clarity):
!python3 run_density_experiment.py --n_seeds 1 --max_cells 22000

# 2. Task 5 validation gate -- confirm the fix actually worked before
#    trusting any of the 3 runs' AUPRC/AUROC numbers:
!python3 check_cell_counts.py
```

Expected `check_cell_counts.py` output: three distinct `n_cells_written`
values (22000/13500/8100) with both PASS lines printed. If either FAILs,
stop and report back rather than treating the numbers as trustworthy.
