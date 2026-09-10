# Claude Code prompt — fix PseudoGRN's max_cells confound (+ acceleration)

Paste into Claude Code, continuing from `pseudogrn/PSEUDOGRN_RESULTS_ANALYSIS.md` and
`CHANGES_max_cells_report.txt`.

---

## Context

Confirmed bug: `--max_cells 3000` is smaller than even Tier 3's natural pool (8,100 cells), so
all three tiers got subsampled down to the identical 3,000 cells before the MI-scoring stage —
same bug class as MTGRN's first sweep. The model result itself is fine (AUPRC 0.34–0.37 vs. 0.078
baseline, 4.4–4.7x, all 9 runs, no failures) — only the density comparison is confounded. This
needs a corrected cap and a re-run, not a re-interpretation of the existing numbers.

**Scope discipline reminder (same as every other model in this project):** `PreprocessData.smooth()`,
`EstimateMI.cal_mi2()`, and `MRMR2()` are frozen third-party code — don't rewrite their algorithm
or logic. Anything you build here is orchestration/acceleration *around* them, same as the
`--max_cells` flag itself already is.

---

## Task 1 — Real timing curve (on actual SERGIO data, not a small stand-in)

The previous cap was extrapolated from a 20-gene/6-TF stand-in dataset and was never checked
against the real 400-gene/37-TF problem — that's part of why this happened. This time, measure
`smooth()` + `cal_mi2()` + `MRMR2()` wall-clock time directly on real SERGIO-derived data at
several concrete pool sizes: 3,000 / 6,000 / 9,000 / 13,500 (Tier 2's natural size), and at least
one or two points beyond that toward Tier 1's natural size (~40,500) if feasible in reasonable
time. Report whether the scaling looks linear, superlinear, or worse — don't assume.

---

## Task 2 — Investigate a legitimate GPU path (don't implement yet)

1. Identify exactly what the nearest-neighbor search inside `cal_mi2()` actually uses (scipy's
   `cKDTree`, sklearn's `NearestNeighbors`, a hand-rolled loop, etc.) — this is the specific
   computational bottleneck per the existing "per-point Python loop" diagnosis.
2. Check whether a GPU-based drop-in replacement exists (e.g. FAISS-GPU, a torch-based kNN) that
   could sit underneath the *same* MI-estimation logic without altering what `cal_mi2()` actually
   computes. This must be a numerical-backend swap only, not a rewrite of the frozen algorithm.
3. **If you find a plausible path, validate it before trusting it:** run both the original CPU
   version and the proposed GPU version on an identical small test case (e.g. 500 cells, a handful
   of candidate pairs) and confirm the MI values match to reasonable numerical precision. Don't
   adopt a "faster but different" result.
4. **If no clean, validated GPU path exists** — plausible, since this sounds like a plain
   Python/scipy loop rather than something naturally GPU-shaped — say so plainly and move to
   Task 3. Don't force a GPU rewrite that touches the frozen estimator's logic just to say GPU was
   used.

---

## Task 3 — CPU parallelization across candidate pairs (safe fallback, build regardless)

The 14,763 TF→target candidate pairs are independent — `cal_mi2()` is called separately per pair.
Parallelize this across CPU cores at the orchestration level (e.g. `multiprocessing.Pool` or
`joblib`), calling the existing frozen `cal_mi2()` once per worker per pair, unmodified. This is
squarely "training orchestration," not a change to frozen code, and doesn't carry Task 2's
correctness risk. Build this regardless of Task 2's outcome — worth having even if GPU also works,
and it's the fallback if GPU doesn't pan out. Re-time Task 1's cost curve with this in place, using
the actual core count of the Kaggle instance you'll run on.

---

## Task 4 — Decide the new cap

Using Task 1/3's real numbers, pick a single `--max_cells` value that sits **above Tier 2's
natural pool (13,500) and below Tier 1's natural pool (~40,500)**. This means Tier 2 and Tier 3
pass through completely uncapped via the existing `min(natural, cap)` logic already in
`sergio_prepare_data.py` — no per-tier special-casing needed, just raising the one value
correctly. If even 13,500 cells turns out too slow for a reasonable session budget, **stop and
report back rather than silently capping Tier 2 too** — that would just reproduce a milder version
of the same confound.

---

## Task 5 — Validation gate (mandatory, before trusting anything downstream)

This bug slipped through once already because nobody checked actual per-run cell counts against
expectations. Before declaring it fixed:
1. Run one (tier, seed) combo per tier — 3 test runs — with the new cap.
2. Explicitly read `n_cells_pooled` / `n_cells_written` back from each run's `prep_meta.json` /
   `metrics.json` and print a direct comparison across the three. Confirm they're genuinely
   different and in the right order: Tier1(capped) > Tier2(natural, 13,500) > Tier3(natural,
   8,100). Don't just trust the code path — show the actual numbers.
3. Only proceed to Task 6 once this explicitly passes.

---

## Task 6 — Re-run and finally analyze

Full 9-run sweep (3 tiers × 3 seeds) on Kaggle with the validated fix. Then aggregate: a summary
CSV (mean±std AUPRC/AUROC per tier, same schema style as Marlene's/RiTINI's), a degradation curve
plot, and an explicit check of whether the per-tier pattern now shows a real trend rather than the
flat-to-inverted signature that originally flagged this bug.

---

## Deliverable

An updated report (extend `PSEUDOGRN_RESULTS_ANALYSIS.md` or a new file) documenting: the timing
curve, the GPU investigation outcome (used or not, and exactly why), the chosen cap and
justification, the Task 5 validation numbers, and the final aggregated, trustworthy PseudoGRN
density results.
