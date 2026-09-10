"""
Orchestrates the temporal-density sparsity experiment (proposal Section 5.5:
Controlled variable: temporal density) across three density tiers x N_SEEDS
random seeds for PseudoGRN, by calling sergio_prepare_data.py +
train_sergio.py as subprocesses for each (tier, seed) combination.

Adapted from marlene/Marlene/run_density_experiment.py -- same tiers, same
seed count, same resumability/logging/--dry_run behavior, same reasoning
for why multiple seeds are needed (see that file's own docstring). Neither
sergio_prepare_data.py nor train_sergio.py is modified by this script -- it
only drives them with the right flags and collects results.

WHY THE SAME TIER DEFINITIONS
------------------------------
TIERS below (15/5/3 out of 15) are identical to Marlene's, and for a given
(tier, seed) our sergio_prepare_data.py's --seed/--n_timepoints_keep draws
the SAME replicate indices Marlene's harness drew for that (tier, seed) --
see sergio_prepare_data.py's module docstring. This is what makes the two
groups' degradation curves (Group A vs Group B, proposal RQ2) comparable
at each nominal tier: both models see data from the identical underlying
SERGIO replicates at a given (tier, seed), just pre-processed differently.

RESUMABILITY
------------
Before running a (tier, seed) combo, this script checks whether
PseudoGRN_results/SERGIO/tier<N>_seed<S>/metrics.json already exists and
skips it if so -- an interrupted sweep (e.g. a Kaggle session timeout) can
simply be re-run from the top.

USAGE
-----
Run from the pseudogrn/ repo root (relative paths for data dirs and results
are resolved against the current working directory, matching where
train_sergio.py itself writes PseudoGRN_results/SERGIO/<runid>/):

    python run_density_experiment.py --n_jobs 4

Sanity-check the sweep plan (prints every command that would run, touches
nothing) before committing compute time:

    python run_density_experiment.py --dry_run

Progress (which tier/seed is running, elapsed time, failures) is both
printed and appended to --log_file (default density_experiment_log.txt),
so progress survives a Kaggle session getting cut off mid-sweep.
"""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "sergio_prepare_data.py"
TRAIN_SCRIPT = SCRIPT_DIR / "train_sergio.py"

# Proposal Section 5.5 temporal-density tiers -- n_timepoints_keep values
# out of the 15 pre-simulated SERGIO replicates loaded as --n_replicates.
# IDENTICAL to marlene/Marlene/run_density_experiment.py's TIERS.
TIERS = {
    1: 15,  # Tier 1 -- dense (all available replicates)
    2: 5,   # Tier 2 -- sparse
    3: 3,   # Tier 3 -- ultra-sparse
}

N_SEEDS = 3  # matches Marlene's sweep -- see module docstring


def log(msg: str, log_file: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_prepare(*, seed, n_timepoints_keep, out_dir, dataset_dir, n_bins,
                 n_replicates, root_bin, max_cells, dry_run, log_file):
    cmd = [
        sys.executable, str(PREPARE_SCRIPT),
        "--dataset_dir", dataset_dir,
        "--n_bins", str(n_bins),
        "--n_replicates", str(n_replicates),
        "--n_timepoints_keep", str(n_timepoints_keep),
        "--seed", str(seed),
        "--root_bin", str(root_bin),
        "--max_cells", str(max_cells),
        "--out_dir", str(out_dir),
    ]
    log(f"  [prepare] {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_train(*, data_dir, runid, window_size, slide, n_jobs, mrmr_lambda,
              seed, dry_run, log_file):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data_dir", str(data_dir),
        "--runid", runid,
        "--window_size", str(window_size),
        "--slide", str(slide),
        "--n_jobs", str(n_jobs),
        "--mrmr_lambda", str(mrmr_lambda),
        "--seed", str(seed),
    ]
    log(f"  [train]   {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="SERGIO bundled dataset folder, forwarded to "
             "sergio_prepare_data.py (same 400-gene/37-TF dataset Marlene "
             "and GENIE3 use).",
    )
    ap.add_argument("--n_bins", type=int, default=9)
    ap.add_argument("--n_replicates", type=int, default=15,
                     help="All 15 pre-simulated replicates are loaded; tiers "
                          "then subsample down to n_timepoints_keep of them "
                          "(matches Marlene's default so seeds align).")
    ap.add_argument("--root_bin", type=int, default=0,
                     help="Forwarded to sergio_prepare_data.py's DPT root "
                          "cell selection.")
    ap.add_argument("--max_cells", type=int, default=22000,
                     help="Forwarded to sergio_prepare_data.py's --max_cells. "
                          "CORRECTED from an earlier 3000 default that "
                          "capped every tier down to the same size, "
                          "destroying the density comparison (see "
                          "PSEUDOGRN_RESULTS_ANALYSIS.md's Fix section) -- "
                          "22000 only compresses Tier 1 (40,500 natural), "
                          "leaving Tier 2/3 (13,500/8,100) at their full "
                          "natural sizes.")
    ap.add_argument("--window_size", type=int, default=5,
                     help="Forwarded to train_sergio.py -- PseudoGRN's own "
                          "main.py default.")
    ap.add_argument("--slide", type=int, default=1,
                     help="Forwarded to train_sergio.py -- PseudoGRN's own "
                          "main.py default.")
    ap.add_argument("--mrmr_lambda", type=float, default=1.0,
                     help="Forwarded to train_sergio.py -- PseudoGRN's own "
                          "mRMR.py default.")
    ap.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1,
                     help="Forwarded to train_sergio.py's --n_jobs (pqdm "
                          "parallelism for cal_mi2/MRMR2, already the "
                          "frozen library's own parallelization mechanism, "
                          "not something built here) -- now defaults to "
                          "the actual detected core count instead of a "
                          "hardcoded guess, per "
                          "pseudogrn_fix_and_reanalyze_prompt.md Task 3.")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--data_root", type=str, default=".",
                     help="Where per-(tier,seed) data_tier<N>_seed<S>/ dirs "
                          "are created.")
    ap.add_argument(
        "--results_root", type=str, default="PseudoGRN_results/SERGIO",
        help="Where train_sergio.py writes <runid>/metrics.json. Must match "
             "train_sergio.py's own hardcoded relative "
             "PseudoGRN_results/SERGIO/<runid> layout -- so run this script "
             "from the same working directory every time (the repo root).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print every command that would run for every (tier, seed) "
             "combo, without executing them or touching the filesystem. "
             "Use this to sanity-check the sweep plan before committing "
             "compute time.",
    )
    ap.add_argument("--log_file", type=str, default="density_experiment_log.txt")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    log_file = Path(args.log_file)

    combos = [
        (tier, n_tp, seed)
        for tier, n_tp in TIERS.items()
        for seed in range(args.n_seeds)
    ]

    log(f"Starting PseudoGRN density sweep: {len(TIERS)} tiers x "
        f"{args.n_seeds} seeds = {len(combos)} runs. dry_run={args.dry_run}",
        log_file)

    n_done = n_skipped = n_failed = 0
    failed_combos = []
    t_sweep_start = time.time()

    for tier, n_tp, seed in combos:
        runid = f"tier{tier}_seed{seed}"
        data_dir = data_root / f"data_tier{tier}_seed{seed}"
        metrics_path = results_root / runid / "metrics.json"

        log(f"=== {runid} (n_timepoints_keep={n_tp}) ===", log_file)

        if metrics_path.exists() and not args.dry_run:
            log(f"  SKIP -- {metrics_path} already exists (resuming)", log_file)
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            run_prepare(
                seed=seed, n_timepoints_keep=n_tp, out_dir=data_dir,
                dataset_dir=args.dataset_dir, n_bins=args.n_bins,
                n_replicates=args.n_replicates, root_bin=args.root_bin,
                max_cells=args.max_cells,
                dry_run=args.dry_run, log_file=log_file,
            )
            run_train(
                data_dir=data_dir, runid=runid,
                window_size=args.window_size, slide=args.slide,
                n_jobs=args.n_jobs, mrmr_lambda=args.mrmr_lambda,
                seed=seed, dry_run=args.dry_run, log_file=log_file,
            )
        except Exception:
            elapsed = time.time() - t0
            log(f"  FAILED after {elapsed / 60:.1f} min:\n{traceback.format_exc()}",
                log_file)
            n_failed += 1
            failed_combos.append(runid)
            continue

        elapsed = time.time() - t0
        log(f"  done in {elapsed / 60:.1f} min", log_file)
        n_done += 1

    total_elapsed = time.time() - t_sweep_start
    log(f"\nSweep finished in {total_elapsed / 60:.1f} min. "
        f"done={n_done} skipped={n_skipped} failed={n_failed} "
        f"(of {len(combos)} total)", log_file)
    if failed_combos:
        log(f"WARNING: failed combo(s): {failed_combos} -- see {log_file} "
            f"for tracebacks. Re-running this script will retry them "
            f"(only combos with an existing metrics.json are skipped).",
            log_file)


if __name__ == "__main__":
    main()
