#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:28:04 2025

@author: linglingzhang
"""
# Replace the Slingshot
import pandas as pd
import numpy as np
import scanpy as sc
import phate
import matplotlib.pyplot as plt
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
import os
os.makedirs("outputtimelag=1", exist_ok=True)
from PreprocessData import smooth_divergence
from psedoScore import cal_mi2_divergence as cal_mutual_information
from mRMR import MRMR2_divergence


#############Getting the csvs using 4 pseudotime methods:
#  Load Data 
df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
adata = sc.AnnData(df_exp)
adata.var_names_make_unique()

#  PHATE 
print("=== Running PHATE ===")
phate_operator = phate.PHATE(n_components=2)
adata.obsm["X_phate"] = phate_operator.fit_transform(adata.X)

# Compute pseudotime as distance from first cell in PHATE space
from sklearn.metrics import pairwise_distances
root_cell_index = 0
distances = pairwise_distances([adata.obsm["X_phate"][root_cell_index]], adata.obsm["X_phate"])[0]
adata.obs["phate_pseudotime"] = distances

# Save PHATE pseudotime
df_phate = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["phate_pseudotime"].values
})
df_phate.to_csv("phate_pseudotime_timlag=2.csv", index=False)

#  Diffusion Maps 
print("=== Running Diffusion Maps ===")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.diffmap(adata)
adata.uns['iroot'] = adata.obsm["X_diffmap"][:, 0].argmin()
sc.tl.dpt(adata)

# Save diffusion pseudotime
df_diff = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["dpt_pseudotime"].values
})
df_diff.to_csv("diffmap_pseudotime_timlag=2.csv", index=False)
print("Saved: diffmap_pseudotime.csv")

#  PCA-based Pseudotime 
print("=== Running PCA-based Pseudotime ===")
pc1 = adata.obsm["X_pca"][:, 0]
pc1_minmax = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
adata.obs["pca_pseudotime"] = pc1_minmax

# Save PCA pseudotime
df_pca = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["pca_pseudotime"].values
})
df_pca.to_csv("pca_pseudotime_timlag=2.csv", index=False)
print("Saved: pca_pseudotime.csv")

#  PAGA 
print("=== Running PAGA ===")
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=1.0)
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, show=False)  # <-- Add this line!
sc.tl.draw_graph(adata, init_pos='paga')
sc.tl.dpt(adata)

# Save PAGA pseudotime
df_paga = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["dpt_pseudotime"].values
})
df_paga.to_csv("paga_pseudotime_timlag=2.csv", index=False)
print("Saved: paga_pseudotime.csv")



####################### We didn't use the example code below in the paper ,since it is no lambda version score
# if you need lambda version score, you can refer the slingshot example code.
###### 5 runs to use 4 pseudotie methods with 3 divergence based score:forward KL, symmetric KL, JS
import pandas as pd
import numpy as np
import os
from EstimateMI import cal_kl2, cal_kl2_symmetric, cal_js2
from mRMR import MRMR2_kl, MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
from PreprocessData import smooth  # use average smoothing

output_dir = "outputtimelag=1_divergence4methods"
os.makedirs(output_dir, exist_ok=True)


def add_noise_to_expression(expression_file: str, output_file: str, noise_level: float = 0.05) -> None:
    """Add Gaussian noise to gene expression data and save."""
    df = pd.read_csv(expression_file, index_col=0)
    ranges = df.max() - df.min()
    noise = np.random.randn(*df.shape) * (ranges.values * noise_level)
    df_noisy = df + noise
    df_noisy[df_noisy < 0] = 0  # clip negative values
    df_noisy.to_csv(output_file)


# Load reference network
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])

# List of pseudotime method CSVs
pseudotime_files = {
    "phate": "phate_pseudotime_timelag=1.csv",
    "pca": "pca_pseudotime_timlag=1.csv",
    "diffmap": "diffmap_pseudotime_timelag=1.csv",
    "paga": "paga_pseudotime_timlag=1.csv"
}

# Divergence-based methods
divergences = {
    "forward_kl": (cal_kl2, MRMR2_kl),
    "symmetric_kl": (cal_kl2_symmetric, MRMR2_divergence),
    "js": (cal_js2, MRMR2_divergence),
}

all_summary = []

for method_name, pse_path in pseudotime_files.items():
    df_pse = pd.read_csv(pse_path, index_col=0)
    print(f"\n=== Pseudotime Method: {method_name} ===")

    run_scores = {div: [] for div in divergences}

    for run in range(1, 6):
        print(f"  >> Run {run}")
        noise_path = f"{output_dir}/ExpressionData_{method_name}_run{run}.csv"
        add_noise_to_expression("input/ExpressionData.csv", noise_path)
        df_exp_noisy = pd.read_csv(noise_path, index_col=0)

        # Use average smoothing (Normi-style)
        count, df_exp_sorted = smooth(df_pse, df_exp_noisy, slipe=1, k=5)

        for name, (score_func, mrmr_func) in divergences.items():
            df_score = score_func(df_exp_sorted, n_jobs=4)
            df_mrmr = mrmr_func(df_score, n_jobs=4)
            df_eval = concat_ref(df_mrmr, df_ref)

            # Clean invalid scores
            df_eval = df_eval[np.isfinite(df_eval['score'])]

            res = cal_auc_aupr(df_eval)
            run_scores[name].append((res["AUROC"], res["AUPRC"]))

            plot_path = f"{output_dir}/network_{method_name}_{name}_run{run}.pdf"
            add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50, output_pdf=plot_path, plot=False)

    # Save mean + individual results
    summary_row = {"Method": method_name}
    for name in divergences:
        aurocs = [s[0] for s in run_scores[name]]
        auprcs = [s[1] for s in run_scores[name]]
        summary_row.update({
            f"{name}_AUROC1": aurocs[0], f"{name}_AUROC2": aurocs[1], f"{name}_AUROC3": aurocs[2],
            f"{name}_AUROC4": aurocs[3], f"{name}_AUROC5": aurocs[4],
            f"{name}_AUROC_mean": np.mean(aurocs),
            f"{name}_AUPRC1": auprcs[0], f"{name}_AUPRC2": auprcs[1], f"{name}_AUPRC3": auprcs[2],
            f"{name}_AUPRC4": auprcs[3], f"{name}_AUPRC5": auprcs[4],
            f"{name}_AUPRC_mean": np.mean(auprcs),
        })
    all_summary.append(summary_row)

# Save final summary
df_summary = pd.DataFrame(all_summary)
df_summary.to_csv(f"{output_dir}/summary_4methods_5runs.csv", index=False)
print("\n=== Finished: summary_4methods_5runs.csv ===")

#############################
#use some other distances based score:
import os
import numpy as np
import pandas as pd
from EstimateMI import cal_backward_kl, cal_wasserstein2, cal_energy2, cal_cramer2
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
from PreprocessData import smooth

def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise based on expression range."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = np.random.randn(*df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0
    return df_noisy

#  Setup 
output_dir = "outputtimelag=1_4distances_4pseudotime"
os.makedirs(output_dir, exist_ok=True)
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])

#  Define divergence methods and function names that match your import 
divergences = {
    "backward_kl": cal_backward_kl,
    "wasserstein": cal_wasserstein2,
    "energy": cal_energy2,
    "cramer": cal_cramer2,
}

# Pseudotime files 
pseudotime_files = {
    "phate": "phate_pseudotime_timelag=1.csv",
    "pca": "pca_pseudotime_timlag=1.csv",
    "diffmap": "diffmap_pseudotime_timlag=1.csv",
    "paga": "paga_pseudotime_timlag=1.csv"
}

# Run experiment 
summary_all = []

for pt_name, pt_path in pseudotime_files.items():
    print(f"\n=== Pseudotime Method: {pt_name} ===")
    df_pse = pd.read_csv(pt_path, index_col=0)

    for run in range(1, 6):
        print(f"  >> Run {run}")
        df_exp_noisy = add_noise(df_exp_orig)
        df_exp_noisy.to_csv(f"{output_dir}/ExpressionData_{pt_name}_run{run}.csv")

        # Step 1: Average smoothing
        _, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)

        run_summary = {"Pseudotime": pt_name, "Run": run}
        for div_name, score_func in divergences.items():
            print(f"     - Score: {div_name}")
            df_score = score_func(df_smoothed, n_jobs=4)
            if df_score.empty:
                print(f"       Skipped {div_name} due to empty score matrix")
                run_summary[f"{div_name}_AUROC"] = np.nan
                run_summary[f"{div_name}_AUPRC"] = np.nan
                continue

            df_mrmr = MRMR2_divergence(df_score, n_jobs=4)
            df_eval = concat_ref(df_mrmr, df_ref)
            df_eval = df_eval[np.isfinite(df_eval["score"])]
            result = cal_auc_aupr(df_eval)
            run_summary[f"{div_name}_AUROC"] = result["AUROC"]
            run_summary[f"{div_name}_AUPRC"] = result["AUPRC"]

            # Save network plot
            plot_path = f"{output_dir}/network_{pt_name}_{div_name}_run{run}.pdf"
            add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                              output_pdf=plot_path, plot=False)

        summary_all.append(run_summary)

#  Save summary table 
df_summary = pd.DataFrame(summary_all)
df_summary.to_csv(f"{output_dir}/summary_4pseudotime_4divergence.csv", index=False)
print("Saved summary to:", f"{output_dir}/summary_4pseudotime_4divergence.csv")

# === Calculate mean and std ===
df_mean = df_summary.groupby("Pseudotime").mean(numeric_only=True).rename_axis("Pseudotime")
df_std = df_summary.groupby("Pseudotime").std(numeric_only=True).rename_axis("Pseudotime")

# Combine and save
df_mean.to_csv(f"{output_dir}/summary_4pseudotime_mean.csv")
df_std.to_csv(f"{output_dir}/summary_4pseudotime_std.csv")
print(" Finished: All runs completed.")
