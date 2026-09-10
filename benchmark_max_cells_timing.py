"""
Task 1 (pseudogrn_fix_and_reanalyze_prompt.md): real timing curve for
smooth() + cal_mi2() + MRMR2() at several pool sizes, measured on REAL
SERGIO-derived data -- not the small 20-gene/6-TF synthetic stand-in the
original --max_cells=3000 default was extrapolated from (see
CHANGES_max_cells_report.txt), which is part of why that default turned
out wrong (below every tier's natural pool size).

Reuses train_sergio.py's own module-level monkeypatch of
PreprocessData.smooth (the pandas-compat fix) by importing train_sergio
directly -- this times the ACTUAL frozen pipeline this project runs, not
a reimplementation of it. No frozen code (PreprocessData.smooth's
patched-in-this-project logic, cal_mi2, MRMR2) is modified here; this
script only calls them at different input sizes and records wall-clock.

DESIGN: DPT/pseudotime only needs to be computed ONCE. Prepare a single
UNCAPPED Tier-1 dataset first (natural 40,500-cell pool, no subsampling),
then this script subsamples DOWN from that one prepared dataset at each
target pool size to time smooth()+cal_mi2()+MRMR2() -- avoiding repeated
expensive DPT computation per pool size tested.

USAGE (Kaggle, two steps)
--------------------------
1. Prepare the uncapped source dataset once:
     python sergio_prepare_data.py \\
       --dataset_dir /kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2 \\
       --n_timepoints_keep 15 --seed 0 --max_cells 0 --out_dir data_timing_source

2. Run this script against it (n_jobs defaults to the detected core count):
     python benchmark_max_cells_timing.py --data_dir data_timing_source

Writes max_cells_timing_curve.csv and prints a table + a per-1000-cells
scaling column (roughly FLAT = linear, GROWING = super-linear -- don't
assume, read the actual numbers).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_sergio  # noqa: E402 -- importing this applies its module-level
# PreprocessData.smooth monkeypatch (pandas-compat fix) as a side effect,
# and gives us load_data/cal_mi2/MRMR2/PreprocessData all from the SAME
# place train_sergio.py itself uses them, not a separate re-import path.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=str, required=True,
                     help="An ALREADY-PREPARED, UNCAPPED sergio_prepare_data.py "
                          "output dir (run with --max_cells 0 first) -- "
                          "pseudotime is reused as-is; this script only "
                          "re-times smooth+cal_mi2+MRMR2 at various "
                          "subsample sizes drawn from it.")
    ap.add_argument("--pool_sizes", type=str,
                     default="3000,6000,9000,13500,20000,27000,40500",
                     help="Comma-separated cell counts to time. Sizes "
                          "larger than the source dataset's actual cell "
                          "count are skipped with a warning, not padded.")
    ap.add_argument("--window_size", type=int, default=5)
    ap.add_argument("--slide", type=int, default=1)
    ap.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1,
                     help="Forwarded to cal_mi2/MRMR2 (pqdm) -- defaults to "
                          "the actual detected core count on whatever "
                          "machine runs this, per Task 3's instruction to "
                          "time with the real Kaggle instance's core count, "
                          "not a guessed number.")
    ap.add_argument("--mrmr_lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="max_cells_timing_curve.csv")
    args = ap.parse_args()

    print(f"n_jobs={args.n_jobs} (os.cpu_count()={os.cpu_count()})")

    data_dir = Path(args.data_dir)
    data_dict = train_sergio.load_data(data_dir)
    df_exp, df_pse = data_dict["df_exp"], data_dict["df_pse"]
    tf_names = data_dict["tf_names"]
    total_cells = df_exp.shape[0]
    print(f"Source dataset: {total_cells} cells x {df_exp.shape[1]} genes (uncapped)")

    pool_sizes = [int(x) for x in args.pool_sizes.split(",")]
    tf_set = set(tf_names)

    rows = []
    for pool_size in pool_sizes:
        if pool_size > total_cells:
            print(f"  [skip] pool_size={pool_size} > available {total_cells} cells")
            continue

        rng = np.random.RandomState(args.seed)
        # Keep indices SORTED after subsampling so pseudotime ordering
        # (baked into row order by sergio_prepare_data.py) survives --
        # smooth() assumes pseudotime-ordered input.
        keep_idx = np.sort(rng.choice(total_cells, size=pool_size, replace=False))
        df_exp_sub = df_exp.iloc[keep_idx]
        df_pse_sub = df_pse.iloc[keep_idx]

        print(f"\n=== pool_size={pool_size} ===")
        t0 = time.time()
        count, df_exp_smooth = train_sergio.PreprocessData.smooth(
            df_pse_sub, df_exp_sub, args.slide, args.window_size,
        )
        t_smooth = time.time() - t0
        print(f"  smooth(): {t_smooth:.1f}s, windows={count}, smoothed shape={df_exp_smooth.shape}")

        t0 = time.time()
        df_mi = train_sergio.cal_mi2(df_exp_smooth, count, n_jobs=args.n_jobs, TF_set=tf_set)
        t_mi = time.time() - t0
        print(f"  cal_mi2(): {t_mi:.1f}s, {df_mi.shape[0]} pairs scored")

        t0 = time.time()
        df_mi_pos = df_mi.loc[df_mi.score > 0].sort_values(by="score", ascending=False)
        df_mrmr = train_sergio.MRMR2(df_mi_pos, n_jobs=args.n_jobs, lambda_val=args.mrmr_lambda)
        t_mrmr = time.time() - t0
        print(f"  MRMR2(): {t_mrmr:.1f}s, {df_mrmr.shape[0]} edges")

        total = t_smooth + t_mi + t_mrmr
        rows.append(dict(
            pool_size=pool_size, n_windows=int(sum(count)),
            smooth_s=round(t_smooth, 1), cal_mi2_s=round(t_mi, 1),
            mrmr_s=round(t_mrmr, 1), total_s=round(total, 1),
            total_min=round(total / 60, 2),
            s_per_1000_cells=round(total / pool_size * 1000, 2),
        ))
        print(f"  TOTAL: {total:.1f}s ({total / 60:.2f} min)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print(df.to_string(index=False))

    if len(df) >= 2:
        print("\nScaling check -- s_per_1000_cells roughly FLAT means "
              "linear scaling; GROWING means super-linear. Read the "
              "actual numbers above, don't assume either.")


if __name__ == "__main__":
    main()
