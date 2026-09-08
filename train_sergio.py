"""
PseudoGRN inference + eval on SERGIO synthetic data.

FROZEN (imported as-is from the vendored Pseudo-network/ clone, never
modified in place -- the two compatibility fixes below live entirely in
THIS file, not in the vendored files; see FIX notes):
    - PreprocessData.smooth   (pseudotime-ordered sliding-window smoothing)
    - EstimateMI.cal_mi2      (Mixed-KSG time-lagged mutual-information
      estimator -- main.py's own STEP2; EstimateMI.py is our thin shim
      re-exporting Pseudo-network/psedoScore.py, see its module docstring
      for why: `EstimateMI.py` was renamed to `psedoScore.py` upstream
      [commit 2d1188c] without updating main.py's import)
    - mRMR.MRMR2              (redundancy-penalized greedy re-ranking --
      main.py's own STEP3)

This script drives exactly main.py's own STEP1 -> STEP2 -> STEP3 sequence
as a library (not a subprocess), with the same arguments main.py's argparse
exposes, so we can pull the intermediate STEP2 score matrix directly for
full-coverage AUPRC/AUROC scoring (main.py itself only ever writes the
final STEP3 rankedEdges.csv to disk).

REWRITTEN for this project (not present in the original repo):
    - load_data(): reads our SERGIO-derived ExpressionData.csv /
      PseudoTime.csv / tf_list.txt / sergio_gt_edges.csv instead of
      whatever BEELINE-style dataset the repo's own examples use.
    - AUPRC/AUROC evaluation against SERGIO ground truth (proposal Section
      5.7), reusing the same "score every candidate edge, feed to
      average_precision_score/roc_auc_score" approach as
      marlene/train_sergio.py::compute_auprc_auroc -- see
      compute_auprc_auroc()'s own docstring below for how the continuous
      score is chosen given PseudoGRN doesn't output a single dense
      attention matrix like Marlene.

FIX -- PreprocessData.smooth() / pandas>=2.0 incompatibility
    `res[item] = ...` inside smooth() (PreprocessData.py) relies on pandas
    Series.__getitem__ falling back to *positional* indexing for an
    integer key not present in a non-integer index. That fallback was
    deprecated for years and is a hard KeyError as of the pandas version
    this project's isolated pseudogrn/ environment pins (verified via a
    minimal repro against the repo's own bundled input/ example data --
    the bug is pre-existing and unrelated to our data, it reproduces on
    ANY input). The fix below is a drop-in replacement of that one line
    (`res[item]` -> `res.iloc[item]`) with IDENTICAL semantics to what the
    code always intended -- not a behavior change, a version-compat patch,
    same class of fix as Marlene's np.in1d shim in marlene/train_sergio.py
    (see that file's own top-of-file comment). We monkeypatch this at
    import time rather than editing Pseudo-network/PreprocessData.py, so
    the vendored clone stays byte-for-byte as cloned.

USAGE
-----
python train_sergio.py \
    --data_dir /kaggle/working/pseudogrn/data \
    --runid pseudogrn_sergio_dense \
    --window_size 5 --slide 1 --n_jobs 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
PSEUDO_NETWORK_DIR = THIS_DIR / "Pseudo-network"
sys.path.insert(0, str(PSEUDO_NETWORK_DIR))

# --- FROZEN imports (see PreprocessData FIX note above for the one
# compatibility patch applied to `smooth` right after this import) --------
import PreprocessData  # noqa: E402
from EstimateMI import cal_mi2  # noqa: E402
from mRMR import MRMR2  # noqa: E402
# ---------------------------------------------------------------------------


def _smooth_pandas_compat(df_p, df_e, slipe, k=5):
    """Drop-in replacement for PreprocessData.smooth -- see module
    docstring FIX note. Every line is unchanged from the original except
    `res[item]` -> `res.iloc[item]`.
    """
    count = []
    for i in range(len(df_p.columns)):
        df_p_c = df_p.iloc[:, i:(i + 1)].dropna()
        df_p_c.columns = ['pseudotime']
        df_p_c.sort_values(by=['pseudotime'], ascending=True, inplace=True)
        df_p_e = pd.merge(df_p_c, df_e, left_index=True, right_index=True)
        df_ee = df_p_e.iloc[:, 1:]

        start, end, j = 0, 0, 0
        df_e_new = pd.DataFrame()
        while end == 0:
            j = j + 1
            if (start + k) < len(df_ee.index):
                df = df_ee.iloc[start:(start + k), :]
            else:
                df = df_ee.iloc[(len(df_ee.index) - k):len(df_ee.index), :]
                end = 1

            res = df.apply(lambda x: x.value_counts().get(0, 0), axis=0).astype(float)
            for item in range(len(df.columns)):
                res.iloc[item] = 0.0 if res.iloc[item] >= k / 2 else df.iloc[:, item].mean()
            df_mean = res.to_frame().T
            df_mean.index = [j]

            df_e_new = df_mean if j == 0 else pd.concat([df_e_new, df_mean])
            start = start + slipe

        df_e_all = df_e_new if i == 0 else pd.concat([df_e_all, df_e_new])
        count.append(len(df_e_new))
    return count, df_e_all


PreprocessData.smooth = _smooth_pandas_compat


def load_data(data_dir: Path):
    """Load our SERGIO-derived ExpressionData.csv / PseudoTime.csv /
    tf_list.txt / sergio_gt_edges.csv. Replaces whatever input-loading the
    repo's own example scripts do -- those hardcode BEELINE example paths,
    irrelevant for our SERGIO-derived data.
    """
    df_exp = pd.read_csv(data_dir / "ExpressionData.csv", index_col=0)
    df_pse = pd.read_csv(data_dir / "PseudoTime.csv", index_col=0)
    tf_names = pd.read_csv(
        data_dir / "tf_list.txt", header=None, index_col=None,
    ).iloc[:, 0].tolist()
    gt_df = pd.read_csv(data_dir / "sergio_gt_edges.csv")
    gt_edges = set(zip(gt_df["regulator"], gt_df["target"]))

    meta_path = data_dir / "prep_meta.json"
    n_timepoints_kept = None
    if meta_path.exists():
        with open(meta_path) as f:
            n_timepoints_kept = json.load(f).get("n_timepoints_kept")

    print(f"(cells, genes) = {df_exp.shape}")
    print(f"n TFs = {len(tf_names)}")
    print(f"n ground-truth edges = {len(gt_edges)}")

    return {
        "df_exp": df_exp, "df_pse": df_pse, "tf_names": tf_names,
        "gt_edges": gt_edges, "n_timepoints_kept": n_timepoints_kept,
    }


def compute_auprc_auroc(df_mi_full, df_mrmr_final, tfs, targets, gt_edges):
    """Primary quantitative metric per proposal Section 5.7. Scores EVERY
    candidate (TF, target) edge against the true edge set, same principle
    as marlene/train_sergio.py::compute_auprc_auroc (which scores Marlene's
    full continuous attention matrix). PseudoGRN has no single dense
    attention-style matrix, so the continuous per-edge confidence is built
    from its own two pipeline stages instead:

    - df_mi_full (STEP2, cal_mi2's raw output) already has COMPLETE
      (TF, target) coverage with score = max(MixedKSG_MI, 0) for every
      candidate pair (cal_mi2 does no filtering internally) -- this is the
      floor/default score for a pair.
    - df_mrmr_final (STEP3, main.py's own score>0 filter followed by
      MRMR2's redundancy-penalized re-ranking) OVERRIDES the default with
      PseudoGRN's own final re-scored confidence wherever it produced one.
      This exactly matches how the repo's OWN example evaluation code
      scores AUPRC/AUROC (see Pseudo-network/run_slingshot_distance_example.py:
      `df_eval = concat_ref(df_mrmr, df_ref); cal_auc_aupr(df_eval)` --
      i.e. the authors' own reported metric is computed on the mRMR
      output, not the raw MI). Using the raw MI score as a floor for pairs
      mRMR's redundancy step or main.py's score>0 filter dropped (rather
      than leaving them out of the evaluation entirely) is the one
      addition on top of the repo's own methodology -- needed because a
      dropped pair from a full-coverage benchmark grid is a real,
      score-appropriate "this model was not confident about this edge"
      data point, not a missing observation; average_precision_score
      requires a score for every candidate to be comparable to Marlene's
      full-matrix scoring.
    """
    score_map = {
        (row.Gene1, row.Gene2): float(row.score) for row in df_mi_full.itertuples()
    }
    for row in df_mrmr_final.itertuples():
        score_map[(row.Gene1, row.Gene2)] = float(row.score)

    y_true, y_score = [], []
    for target in targets:
        for tf in tfs:
            if tf == target:
                continue
            y_true.append(1 if (tf, target) in gt_edges else 0)
            y_score.append(score_map.get((tf, target), 0.0))
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_true.sum() == 0:
        return float("nan"), float("nan")
    return average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/kaggle/working/pseudogrn/data")
    ap.add_argument("--runid", type=str, default="pseudogrn_sergio_run")
    ap.add_argument("--device", type=str, default="cpu",
                     help="Unused by PseudoGRN itself (CPU-only estimator, "
                          "no torch/GPU code anywhere in the repo) -- kept "
                          "only so run_density_experiment.py's CLI mirrors "
                          "Marlene's and GENIE3's shape.")
    ap.add_argument("--window_size", type=int, default=5,
                     help="Sliding-window size for smooth(). PseudoGRN's "
                          "own main.py default -- not tuned by us.")
    ap.add_argument("--slide", type=int, default=1,
                     help="Sliding-window step for smooth(). PseudoGRN's "
                          "own main.py default.")
    ap.add_argument("--n_jobs", type=int, default=1,
                     help="Parallel workers for cal_mi2/MRMR2 (pqdm). "
                          "PseudoGRN's own main.py default is 1; raise this "
                          "on a multi-core Kaggle instance for speed, it "
                          "does not change results.")
    ap.add_argument("--mrmr_lambda", type=float, default=1.0,
                     help="mRMR redundancy-penalty weight. PseudoGRN's own "
                          "mRMR.py default (both MRMR() and MRMR2() default "
                          "to lambda_val=1.0) -- not tuned by us.")
    ap.add_argument("--seed", type=int, default=0,
                     help="Random seed (numpy). PseudoGRN's own pipeline "
                          "(smooth -> cal_mi2 -> MRMR2) is deterministic "
                          "given its inputs -- there is no weight init or "
                          "batch sampling to seed, unlike Marlene. Kept for "
                          "interface parity with run_density_experiment.py "
                          "and to seed sergio_prepare_data.py's own tier "
                          "subsampling upstream of this script.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    print(f"Initializing run '{args.runid}'")
    data_dict = load_data(data_dir)
    df_exp, df_pse = data_dict["df_exp"], data_dict["df_pse"]
    tf_names, gt_edges = data_dict["tf_names"], data_dict["gt_edges"]
    all_genes = df_exp.columns.tolist()

    if len(tf_names) < 2:
        raise ValueError(
            f"n_TFs={len(tf_names)}, too few regulators to form a "
            "meaningful candidate edge set. Check sergio_prepare_data.py's "
            "TF detection against gt_GRN.csv for this dataset."
        )

    # --- STEP1: pseudotime-ordered sliding-window smoothing (frozen, patched) --
    count, df_exp_smooth = PreprocessData.smooth(
        df_pse, df_exp, args.slide, args.window_size,
    )
    print(f"smooth(): {len(count)} branch(es), windows={count}, "
          f"smoothed shape={df_exp_smooth.shape}")

    # --- STEP2: Mixed-KSG time-lagged mutual information (frozen) -------------
    tf_set = set(tf_names)
    df_mi = cal_mi2(df_exp_smooth, count, n_jobs=args.n_jobs, TF_set=tf_set)
    print(f"cal_mi2(): {df_mi.shape[0]} candidate (TF, target) pairs scored")

    # --- STEP3: main.py's own score>0 filter + MRMR2 redundancy re-ranking ----
    df_mi_pos = df_mi.loc[df_mi.score > 0].sort_values(by="score", ascending=False)
    df_mrmr = MRMR2(df_mi_pos, n_jobs=args.n_jobs, lambda_val=args.mrmr_lambda)
    df_mrmr_final = df_mrmr.sort_values(by="score", ascending=False)
    print(f"MRMR2(): {df_mrmr_final.shape[0]} re-ranked edges")

    # --- Evaluation -------------------------------------------------------
    auprc, auroc = compute_auprc_auroc(df_mi, df_mrmr_final, tf_names, all_genes, gt_edges)
    print(f"\nAUPRC={auprc:.4f} AUROC={auroc:.4f}")

    out_dir = Path("PseudoGRN_results") / "SERGIO" / args.runid
    out_dir.mkdir(parents=True, exist_ok=True)

    df_mrmr_out = df_mrmr_final.loc[df_mrmr_final.score > 0]
    df_mrmr_out.to_csv(out_dir / "rankedEdges.csv", header=True, index=False)

    n_timepoints = data_dict["n_timepoints_kept"]
    if n_timepoints is None:
        print("WARNING: data_dir has no prep_meta.json (n_timepoints_kept "
              "unknown) -- was this data prepared by an older "
              "sergio_prepare_data.py? Falling back to n_cells as a "
              "non-comparable placeholder.")
        n_timepoints = int(df_exp.shape[0])
    metrics = {
        "mean_auprc": float(auprc),
        "mean_auroc": float(auroc),
        # Group B produces one prediction over the whole pooled pseudotime
        # trajectory, not one per discrete timepoint like Marlene's
        # per-celltype loop -- see sergio_prepare_data.py's module
        # docstring. Single-element lists kept for structural parity with
        # Marlene's metrics.json (make_comparison_outputs.py-style
        # aggregation only reads mean_auprc/mean_auroc/n_timepoints).
        "auprc_per_t": [float(auprc)],
        "auroc_per_t": [float(auroc)],
        "n_timepoints": n_timepoints,
        "n_cells_pooled": int(df_exp.shape[0]),
        "n_genes": int(df_exp.shape[1]),
        "n_tfs": len(tf_names),
        "n_candidate_edges_scored": int(df_mi.shape[0]),
        "n_edges_after_mrmr": int(df_mrmr_out.shape[0]),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved rankedEdges.csv + metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
