"""
Converts a SERGIO-simulated dataset into the raw single-cell + pseudotime
input format PseudoGRN's (frozen, unmodified) main.py pipeline expects --
this is Group B (continuous/pseudotime) per the proposal, so it is NOT a
copy of Marlene's sergio_prepare_data.py despite loading the same source
files.

WHY THIS SCRIPT LOOKS DIFFERENT FROM MARLENE'S
------------------------------------------------
Group A (Marlene) consumes discrete, labeled snapshots: SERGIO bin id ->
cell_type, SERGIO replicate id -> timepoint, both passed to the model as
explicit labels. Group B (PseudoGRN) consumes raw single-cell data with
real timestamps and computes its OWN internal pseudotime ordering -- it
never sees a replicate or bin label at all. Concretely:

  - We reuse Marlene's SERGIO-loading structure (same
    `simulated_noNoise_{rep}.csv` + `gt_GRN.csv` format, same per-bin
    column-block layout) to get raw counts + bin ids + gene ids.
  - We reuse Marlene's --n_timepoints_keep / --seed density-tier
    subsampling MECHANISM VERBATIM (identical
    `np.random.RandomState(seed).choice(n_replicates, size=k,
    replace=False)` call, same tier definitions 15/5/3 out of 15) so that,
    for a given (tier, seed), this script keeps the SAME replicate indices
    Marlene's harness kept for that (tier, seed) -- the two harnesses never
    call any data-dependent code before this RNG draw, so identical
    (seed, n_replicates, n_timepoints_keep) inputs give identical output
    regardless of which script calls it.
  - Once the replicate subset is fixed, we POOL all raw cells from the
    kept replicates into one unlabeled cell set and DISCARD the replicate
    id entirely -- it is never written to any output file PseudoGRN reads.
    Bin id is kept internally ONLY to pick a deterministic DPT root cell
    (see --root_bin below); it is likewise never written to PseudoGRN's
    input files.
  - Pseudotime is computed over this pooled, unlabeled cell set using
    scanpy's diffusion pseudotime pipeline (sc.tl.pca -> sc.tl.diffmap ->
    sc.tl.dpt), seeded from a root cell in --root_bin (default: bin 0,
    i.e. the first SERGIO cell type/state -- an arbitrary but deterministic
    and documented anchor; DPT's *ordering* is a property of the diffusion
    manifold, not of which specific cell within the root region is chosen,
    so this choice affects reproducibility, not correctness).

Temporal density is controlled at the level of DATA AVAILABILITY, not
label format: for a given (tier, seed), this script restricts to the same
subset of SERGIO replicates Marlene's harness used for that (tier, seed).
Marlene consumes these as discrete labeled snapshots; here they are pooled
and consumed as an unlabeled cell set, over which pseudotime is computed
internally, per PseudoGRN's native (Group B) input format. This mirrors
the proposal's own Section 5.2 framing (both groups see the same
underlying biology, differing only in required pre-processing) and follows
established practice in the trajectory-inference literature -- Saelens et
al. (Nature Biotechnology, 2019; the dynverse/dynbenchmark trajectory-
inference benchmark) evaluate pseudotime method robustness via cell/data
subsampling as one of their core benchmark axes, which is the direct
precedent for this design. This is cited alongside the existing BEELINE
reference in the methodology note -- this is standard benchmarking
practice for pseudotime methods, not a workaround.

PREPROCESSING
--------------
Applies the same sc.pp.normalize_total + sc.pp.log1p Marlene used, for
parity across models per the proposal's fair-comparison protocol (Section
5.6). PseudoGRN's own repo (README.md, main.py, PreprocessData.py) does
not document any required input normalization -- ExpressionData.csv is
read as a plain numeric matrix with no preprocessing step of its own -- so
there is no documented PseudoGRN-specific preprocessing to defer to, and
normalize_total+log1p is applied to match Marlene (same reasoning as
Marlene's own v4 fix: SERGIO bins differ hugely in raw expression SCALE
by design, which would otherwise dominate any per-gene distance/MI score).

OUTPUT (PseudoGRN / BEELINE-style input format, confirmed against this
repo's bundled input/ example and main.py's own argparse):
- ExpressionData.csv : cells (rows) x genes (columns), index=cell id
- PseudoTime.csv     : cells (rows) x 1 pseudotime column, index=cell id
                       (single column "PseudoTime1" -- SERGIO's pooled
                       replicates form one lineage/branch here, unlike
                       Slingshot's multi-branch output with per-branch
                       NaNs, so no branching structure is introduced)
- tf_list.txt        : one gene name per line (for main.py's --tf_file)
- sergio_gt_edges.csv: regulator,target -- OUR OWN evaluation ground
                       truth (never passed to PseudoGRN itself; used only
                       by train_sergio.py's AUPRC/AUROC scoring, same role
                       as Marlene's sergio_gt_edges.csv)
"""
import argparse
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def load_sergio_dataset(
    dataset_dir: Path,
    n_bins: int,
    replicate_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Returns X, cell_type (=bin id), replicate (=SERGIO replicate id),
    gene_ids, gt_edges. Identical loading logic to Marlene's
    sergio_prepare_data.py::load_sergio_dataset (same file format), kept
    separate here (not imported) so the two harnesses stay independently
    readable/runnable, matching the existing marlene/ vs genie3/ split.
    """
    gt_path = dataset_dir / "gt_GRN.csv"
    gt_df = pd.read_csv(gt_path, header=None, names=["reg", "target"])
    gt_edges = list(zip(gt_df["reg"].astype(int), gt_df["target"].astype(int)))

    X_list, CT_list, REP_list = [], [], []
    gene_ids = None

    for rep in replicate_ids:
        csv_path = dataset_dir / f"simulated_noNoise_{rep}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found")
            continue

        raw = pd.read_csv(csv_path, header=None)
        gene_ids_this = raw.iloc[1:, 0].to_numpy().astype(int)
        expr = raw.iloc[1:, 1:].to_numpy().astype(np.float32)  # (n_genes, n_cells)

        if gene_ids is None:
            gene_ids = gene_ids_this
        else:
            assert np.array_equal(gene_ids, gene_ids_this), \
                "gene order mismatch between replicates"

        n_genes, n_cells_total = expr.shape
        assert n_cells_total % n_bins == 0, \
            f"{n_cells_total} cells not divisible by {n_bins} bins"
        cells_per_bin = n_cells_total // n_bins

        cell_type = np.repeat(np.arange(n_bins), cells_per_bin)
        replicate = np.full(n_cells_total, rep, dtype=int)

        X_list.append(expr.T)  # -> (n_cells, n_genes)
        CT_list.append(cell_type)
        REP_list.append(replicate)
        print(f"  loaded {csv_path.name} (replicate={rep}): "
              f"{expr.shape[1]} cells x {n_genes} genes, {n_bins} cell types")

    X = np.concatenate(X_list, axis=0)
    cell_type = np.concatenate(CT_list, axis=0)
    replicate = np.concatenate(REP_list, axis=0)
    return X, cell_type, replicate, gene_ids, gt_edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="Path to a SERGIO bundled dataset folder (same dataset Marlene "
             "and GENIE3 use, for a like-for-like comparison).",
    )
    ap.add_argument("--n_bins", type=int, default=9,
                     help="Number of SERGIO bins/cell-types in this dataset "
                          "(9 for DS2). Used only to pick the DPT root cell.")
    ap.add_argument(
        "--n_replicates", type=int, default=15,
        help="How many of the pre-simulated replicate runs to load before "
             "tier subsampling (15 = all of them, matching "
             "run_density_experiment.py's default so the same --seed "
             "reproduces Marlene's exact replicate selection).",
    )
    ap.add_argument(
        "--n_timepoints_keep", type=int, default=None,
        help="If set, keep only this many of --n_replicates replicates "
             "(density-tier ablation) -- SAME MECHANISM as Marlene's "
             "sergio_prepare_data.py: a --seed-controlled "
             "np.random.RandomState(seed).choice(...) draw, so a given "
             "(tier, seed) selects the identical replicate indices Marlene "
             "used. Unlike Marlene, the kept replicates are POOLED into one "
             "unlabeled cell set afterward -- see module docstring.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed controlling which replicates --n_timepoints_keep "
             "subsamples. Must match Marlene's --seed for the same (tier, "
             "seed) combo to guarantee identical replicate selection.",
    )
    ap.add_argument(
        "--root_bin", type=int, default=0,
        help="SERGIO bin id whose cells seed the DPT root (diffusion "
             "pseudotime needs a root cell). Bin id is otherwise discarded "
             "after this choice -- PseudoGRN never sees it. Arbitrary but "
             "deterministic; see module docstring.",
    )
    ap.add_argument(
        "--max_cells", type=int, default=3000,
        help="Cap on pooled cells actually WRITTEN for PseudoGRN to consume "
             "(DPT pseudotime is still computed over the full pooled pool "
             "first, for manifold quality -- this only subsamples the "
             "output). NEEDED because main.py's smooth() (window_size=5, "
             "slide=1) barely reduces cell count into windows, and its "
             "Mixed-KSG MI estimator (cal_mi2) has a pure-Python per-point "
             "loop that scales worse than linearly in window count -- "
             "measured on the real 400-gene/37-TF SERGIO dataset: ~2s per "
             "candidate (TF, target) pair at N=40496 windows, i.e. "
             "~8+ hours for ONE tier-1 run's 14763 pairs. PseudoGRN's own "
             "repo was evidently built/tested at BEELINE scale (~2000 "
             "cells), not SERGIO's pooled 8k-40k cells. 3000 was chosen from "
             "a direct timing of smooth()+cal_mi2()+MRMR2() at N=2996 "
             "windows (on a 20-gene/6-TF synthetic stand-in, not the real "
             "dataset -- no real SERGIO data was available in the "
             "environment this was built in), scaled to 400 genes/37 "
             "TFs/14763 pairs by each stage's own measured per-gene / "
             "per-pair cost: ~5 min smooth() + ~10 min cal_mi2() + ~3 min "
             "MRMR2() with --n_jobs 4, roughly 20-25 min per (tier, seed) "
             "run, ~3-4 hours for the full 9-run sweep. TREAT THIS AS AN "
             "ESTIMATE, not a verified real-data number -- if your first "
             "real run's timing differs noticeably, that's a signal to "
             "raise or lower --max_cells accordingly. This is independent of "
             "--n_timepoints_keep, so all three density tiers get the same "
             "compute budget -- "
             "the tier variable stays 'how many SERGIO replicates were "
             "available', not 'how many cells fed the estimator'. Raise "
             "this only if you have verified wall-clock budget for it; set "
             "to 0 to disable capping entirely (not recommended above "
             "~5000 pooled cells).",
    )
    ap.add_argument("--out_dir", type=str, default="/kaggle/working/pseudogrn/data")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SERGIO dataset from {dataset_dir}")
    X, cell_type, replicate, gene_ids, gt_edges = load_sergio_dataset(
        dataset_dir, n_bins=args.n_bins,
        replicate_ids=list(range(args.n_replicates)),
    )
    print(f"Total: {X.shape[0]} cells x {X.shape[1]} genes, "
          f"{len(gt_edges)} ground-truth edges, "
          f"{len(np.unique(cell_type))} cell types, "
          f"{len(np.unique(replicate))} replicates loaded")

    gene_names = np.array([f"G{gid}" for gid in gene_ids])
    id_to_name = {gid: f"G{gid}" for gid in gene_ids}

    # --- density-tier subsampling: IDENTICAL mechanism/call to Marlene's,
    # so the same (seed, n_replicates, n_timepoints_keep) picks the same
    # replicate indices (see module docstring). Runs before pooling.
    if args.n_timepoints_keep is not None:
        k = min(args.n_timepoints_keep, args.n_replicates)
        rng = np.random.RandomState(args.seed)
        keep_rep = np.sort(rng.choice(args.n_replicates, size=k, replace=False))
        mask = np.isin(replicate, keep_rep)
        X, cell_type, replicate = X[mask], cell_type[mask], replicate[mask]
        print(f"Subsampled to {len(keep_rep)} replicates (seed={args.seed}, "
              f"replicates {sorted(keep_rep)})")

    # --- pool: discard replicate id entirely from here on (Group B never
    # sees it). cell_type (bin id) is retained ONLY to select the DPT root.
    print(f"Pooled cell set for PseudoGRN: {X.shape[0]} cells x {X.shape[1]} genes "
          f"(replicate id discarded, per Group B native input format)")

    regulators = set(r for r, _ in gt_edges)
    is_tf = np.array([gid in regulators for gid in gene_ids])
    print(f"{is_tf.sum()} / {len(gene_ids)} genes are regulators (TFs)")
    if is_tf.sum() < 2:
        print("WARNING: fewer than 2 TFs -- candidate edge set will be "
              "degenerate.")

    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": np.array([f"CT{c}" for c in cell_type])}),
        var=pd.DataFrame({"is_TF": is_tf}, index=gene_names),
    )
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]

    # Parity preprocessing with Marlene -- see module docstring PREPROCESSING
    # section for why (SERGIO bins differ in raw expression scale by design;
    # PseudoGRN's own repo documents no input normalization of its own).
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # --- Diffusion pseudotime over the pooled, unlabeled cell set ---------
    n_pcs = min(30, adata.n_vars - 1, adata.n_obs - 1)
    n_neighbors = min(15, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=max(2, n_pcs))
    sc.pp.neighbors(adata, n_neighbors=max(2, n_neighbors))
    sc.tl.diffmap(adata)

    root_mask = cell_type == args.root_bin
    if not root_mask.any():
        print(f"WARNING: --root_bin={args.root_bin} not present in pooled "
              f"cells; falling back to bin {int(cell_type.min())}")
        root_mask = cell_type == cell_type.min()
    root_idx = int(np.flatnonzero(root_mask)[0])
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    print(f"Computed DPT pseudotime, root cell index={root_idx} "
          f"(bin {args.root_bin})")

    # --- Cap cell count for compute tractability (see --max_cells help) ---
    # Pseudotime was computed over the FULL pool above (manifold quality);
    # this subsampling only affects what's written for PreprocessData.smooth
    # + cal_mi2 to actually run on downstream. Seeded by --seed for
    # reproducibility (same seed -> same subsample, distinct from -- and
    # independent of -- the replicate-level tier subsampling seed use above).
    n_full_pool = adata.n_obs
    if args.max_cells and adata.n_obs > args.max_cells:
        sub_rng = np.random.RandomState(args.seed)
        keep_idx = np.sort(sub_rng.choice(adata.n_obs, size=args.max_cells, replace=False))
        adata = adata[keep_idx].copy()
        print(f"Capped pooled cells for compute tractability: "
              f"{n_full_pool} -> {adata.n_obs} (--max_cells={args.max_cells}, "
              f"seed={args.seed}); pseudotime was computed before this cap")

    # --- Write PseudoGRN-native input files --------------------------------
    expr_df = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names,
    )
    expr_path = out_dir / "ExpressionData.csv"
    expr_df.to_csv(expr_path)
    print(f"Wrote {expr_path}  ({expr_df.shape[0]} cells x {expr_df.shape[1]} genes)")

    pt_df = pd.DataFrame(
        {"PseudoTime1": adata.obs["dpt_pseudotime"].to_numpy()},
        index=adata.obs_names,
    )
    pt_path = out_dir / "PseudoTime.csv"
    pt_df.to_csv(pt_path)
    print(f"Wrote {pt_path}")

    tf_names = gene_names[is_tf]
    tf_path = out_dir / "tf_list.txt"
    pd.Series(tf_names).to_csv(tf_path, header=False, index=False)
    print(f"Wrote {tf_path}  ({len(tf_names)} TFs)")

    gt_edges_named = [(id_to_name[r], id_to_name[t]) for r, t in gt_edges
                       if r in id_to_name and t in id_to_name]
    gt_path = out_dir / "sergio_gt_edges.csv"
    pd.DataFrame(gt_edges_named, columns=["regulator", "target"]).to_csv(
        gt_path, index=False
    )
    print(f"Wrote {gt_path}  ({len(gt_edges_named)} edges)")

    # n_timepoints_kept (number of SERGIO replicates pooled into this cell
    # set) is the proposal's Section 5.5 density-tier axis -- same quantity
    # Marlene's h5ad encodes via its (now-discarded) timepoint label. We no
    # longer have that label on the pooled cell set by design (see module
    # docstring), so train_sergio.py can't recover it from the data alone;
    # write it out explicitly so metrics.json can report the same x-axis
    # make_comparison_outputs.py plots for Marlene/GENIE3.
    n_replicates_kept = (
        len(keep_rep) if args.n_timepoints_keep is not None else args.n_replicates
    )
    meta_path = out_dir / "prep_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_timepoints_kept": int(n_replicates_kept),
            "seed": args.seed,
            "n_cells_pooled_full": int(n_full_pool),
            "n_cells_written": int(adata.n_obs),
            "max_cells_cap": args.max_cells,
        }, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
