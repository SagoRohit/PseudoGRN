
############################# All example using the slingshot
########################using different lambda for all distances:
import os
import numpy as np
import pandas as pd
from PreprocessData import smooth
from psedoScore import (
    cal_kl2, cal_kl2_symmetric, cal_js2,
    cal_backward_kl, cal_wasserstein2, cal_energy2, cal_cramer2,
    cal_pearson2, cal_symmetric_pearson2, cal_js_pearson2, cal_neyman2
)
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot


def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise to expression matrix."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = np.random.randn(*df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0
    return df_noisy

#  Input Paths 
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
output_dir = "output_slingshot_divergence10"
os.makedirs(output_dir, exist_ok=True)

#  Divergence Methods 
divergences = {
    "forwardKL": cal_kl2,
    "symmetricKL": cal_kl2_symmetric,
    "JS": cal_js2,
    "backwardKL": cal_backward_kl,
    "wasserstein": cal_wasserstein2,
    "energy": cal_energy2,
    "cramer": cal_cramer2,
    "pearson": cal_pearson2,
    "symmetric_pearson": cal_symmetric_pearson2,
    "js_pearson": cal_js_pearson2,
    "neyman": cal_neyman2,
}

#  Main Loop 
summary_all = []

lambda_vals = [0.1, 1, 1.5, 2, 2.5, 3, 5]  

for run in range(1, 6):
    print(f"\n=== RUN {run} ===")
    df_exp_noisy = add_noise(df_exp_orig)
    count, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)
    run_summary = {"Run": run}

    for name, score_func in divergences.items():
        print(f"  >> {name}")
        df_score = score_func(df_smoothed, n_jobs=4)

        # Reshape matrix to long form
        if set(df_score.columns) == set(df_smoothed.columns):
            df_score = df_score.stack().reset_index()
            df_score.columns = ['Gene1', 'Gene2', 'score']

        df_score['score'] = pd.to_numeric(df_score['score'], errors='coerce')

        for lambda_val in lambda_vals:
            print(f"     -- λ = {lambda_val}")
            # Filter top-score edges (simulate sparsity)
            #df_score_lambda = df_score.copy()
            #threshold = df_score_lambda['score'].quantile(1 - lambda_val)
            #df_score_lambda = df_score_lambda[df_score_lambda['score'] >= threshold]
            df_mrmr = MRMR2_divergence(df_score, n_jobs=4, lambda_val=lambda_val)

            #df_mrmr = MRMR2_divergence(df_score_lambda, n_jobs=4)
            df_eval = concat_ref(df_mrmr, df_ref)
            df_eval = df_eval[np.isfinite(df_eval["score"])]

            result = cal_auc_aupr(df_eval)
            run_summary[f"{name}_λ{lambda_val}_AUROC"] = result["AUROC"]
            run_summary[f"{name}_λ{lambda_val}_AUPRC"] = result["AUPRC"]

            plot_path = f"{output_dir}/network_{name}_lambda{lambda_val}_run{run}.pdf"
            try:
                add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                                  output_pdf=plot_path, plot=False)
                print(f"       [Plot Saved] {plot_path}")
            except Exception as e:
                print(f"       [Plot Error] {e}")

    summary_all.append(run_summary)

# Save Summary in Wide Format with All Runs
def save_results_wide_format_fixed(df_summary, output_dir):
    import os
    import pandas as pd

    # Convert long-style column names into parts
    df_long = pd.melt(
        df_summary,
        id_vars=['Run'],
        var_name='metric_lambda',
        value_name='score'
    )

    # Extract: distance, lambda, metric (AUROC/AUPRC)
    components = df_long['metric_lambda'].str.extract(
        r'(?P<distance>[a-zA-Z0-9]+)_λ(?P<lambda>[0-9.]+)_(?P<metric>AUROC|AUPRC)'
    )
    df_long = pd.concat([df_long, components], axis=1).dropna()

    # Convert lambda to float for sorting
    df_long['lambda'] = df_long['lambda'].astype(float)

    # Pivot to put AUROC and AUPRC as columns, with runs spread across rows
    df_pivot = df_long.pivot_table(
        index=['distance', 'lambda', 'Run'],
        columns='metric',
        values='score'
    ).reset_index()

    # Group by distance and lambda, then collect 5 runs per metric
    grouped = df_pivot.groupby(['distance', 'lambda'])

    output_rows = []
    for (dist, lam), group in grouped:
        row = {
            'pseudotime_method': 'Slingshot',
            'distance': dist,
            'lambda': lam
        }

        for i, r in enumerate(group.itertuples(), 1):
            row[f'AUROC_run{i}'] = r.AUROC
            row[f'AUPRC_run{i}'] = r.AUPRC

        # Add mean values
        row['AUROC_mean'] = group['AUROC'].mean()
        row['AUPRC_mean'] = group['AUPRC'].mean()
        output_rows.append(row)

    df_out = pd.DataFrame(output_rows)

    # Column order
    column_order = ['pseudotime_method', 'distance', 'lambda']
    for metric in ['AUROC', 'AUPRC']:
        column_order += [f"{metric}_run{i}" for i in range(1, 6)]
        column_order += [f"{metric}_mean"]

    df_out = df_out[column_order]

    # Save
    output_path = os.path.join(output_dir, "results_wide_format.csv")
    df_out.to_csv(output_path, index=False)
    print(f"Saved corrected wide format to:\n{output_path}")
    return df_out
'''
#another version to save the summary csv:
df_summary = pd.DataFrame(summary_all)
summary_path = os.path.join(output_dir, "summary_divergence_results.csv")
df_summary.to_csv(summary_path, index=False)
'''

################# we can use cross validation to find the best lambda for the auc of using cramer and plot the curve:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from PreprocessData import smooth
from psedoScore import cal_cramer2
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr


output_dir = "output_cramer"
os.makedirs(output_dir, exist_ok=True)


df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])


_, df_smoothed = smooth(df_pse, df_exp, slipe=1, k=5)


lambda_vals = [0.1, 0.5, 1, 1.5, 2, 3, 5]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
results = []


for lam in lambda_vals:
    aucs = []
    for train_idx, test_idx in kf.split(df_ref):
        ref_train = df_ref.iloc[train_idx]
        ref_test = df_ref.iloc[test_idx]

        df_score = cal_cramer2(df_smoothed, n_jobs=4)
        if set(df_score.columns) == set(df_smoothed.columns):
            df_score = df_score.stack().reset_index()
            df_score.columns = ['Gene1', 'Gene2', 'score']
        df_score['score'] = pd.to_numeric(df_score['score'], errors='coerce')

        df_mrmr = MRMR2_divergence(df_score, n_jobs=4, lambda_val=lam)
        df_eval = concat_ref(df_mrmr, ref_test)
        df_eval = df_eval[np.isfinite(df_eval["score"])]
        auc = cal_auc_aupr(df_eval)["AUROC"]
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    results.append((lam, mean_auc))


df_results = pd.DataFrame(results, columns=["lambda", "mean_AUROC"])
df_results.to_csv(f"{output_dir}/cv_results_cramer.csv", index=False)


plt.figure()
plt.plot(df_results["lambda"], df_results["mean_AUROC"], marker="o")
plt.xlabel("Lambda")
plt.ylabel("Mean AUROC")
plt.title("5-fold CV: AUROC vs Lambda (Cramér)")
plt.grid(True)
plt.savefig(f"{output_dir}/cv_auc_vs_lambda_cramer.pdf")
plt.close()

#################################### plot the boxplot of auc and prc:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV files which we got from above and run_4new pseudo_methods_example.py

df1 = pd.read_csv("summary_4sepudotime methods mean.csv")
df2 = pd.read_csv("summary_slingshot_lambda1.5.csv")

# Mapping for divergence name formatting and short names
name_map = {
    'forwardKL': ('Forward KL', 'F-KL'),
    'symmetricKL': ('Symmetric KL', 'S-KL'),
    'JS': ('JS', 'JS'),
    'backwardKL': ('Backward KL', 'B-KL'),
    'pearson': ('Pearson', 'Pearson'),
    'symmetric_pearson': ('Symmetric Pearson', 'S-Pearson'),
    'js_pearson': ('JS Pearson', 'JS-Pearson'),
    'neyman': ('Neyman', 'Neyman'),
    'wasserstein': ('Wasserstein', 'Wass'),
    'energy': ('Energy', 'Energy'),
    'cramer': ('Cramér', 'Cramer')
}

# Define plot order using the keys from name_map (excluding backwardKL and js_pearson)
ordered_keys = ['forwardKL', 'symmetricKL', 'JS', 'pearson', 'symmetric_pearson', 'neyman', 'wasserstein', 'energy', 'cramer']
ordered_names = [name_map[key][0] for key in ordered_keys]  # Full names
short_names = [name_map[key][1] for key in ordered_keys]    # Short names

# Pseudotime methods
pseudotime_methods = ['diffmap', 'paga', 'pca', 'phate']

# Collect AUROC data
auroc_data = []
for raw in name_map.keys():
    if raw not in ordered_keys:  # Skip backwardKL and js_pearson
        continue
    vals = []
    # Add values from df1 (diffmap, paga, pca, phate)
    for method in pseudotime_methods:
        value = df1[(df1['Divergence'] == raw) & (df1['Pseudotime'] == method)]['AUROC'].values
        if len(value) > 0:
            vals.append(float(value[0]))
    # Add value from df2 (slingshot)
    value = df2[df2['Divergence'] == raw]['AUROC (Mean ± SD)'].str.split(' ± ').str[0].astype(float).values
    if len(value) > 0:
        vals.append(float(value[0]))
    if vals:
        auroc_data.append(vals)


label_to_data = dict(zip(ordered_names, auroc_data))
sorted_auroc_data = [label_to_data[name] for name in ordered_names]

# Collect AUPRC data
auprc_data = []
for raw in name_map.keys():
    if raw not in ordered_keys:  # Skip backwardKL and js_pearson
        continue
    vals = []
    # Add values from df1 (diffmap, paga, pca, phate)
    for method in pseudotime_methods:
        value = df1[(df1['Divergence'] == raw) & (df1['Pseudotime'] == method)]['AUPRC'].values
        if len(value) > 0:
            vals.append(float(value[0]))
    # Add value from df2 (slingshot)
    value = df2[df2['Divergence'] == raw]['AUPRC (Mean ± SD)'].str.split(' ± ').str[0].astype(float).values
    if len(value) > 0:
        vals.append(float(value[0]))
    if vals:
        auprc_data.append(vals)

# Ensure order is correct
label_to_data = dict(zip(ordered_names, auprc_data))
sorted_auprc_data = [label_to_data[name] for name in ordered_names]

# Plot AUROC
fig1, ax1 = plt.subplots(figsize=(12, 8))
bp1 = ax1.boxplot(sorted_auroc_data, patch_artist=True, labels=short_names)
for patch in bp1['boxes']:
    patch.set_facecolor('white')  # Remove color inside boxes
for whisker in bp1['whiskers']:
    whisker.set(color='black', linewidth=1.5)
for cap in bp1['caps']:
    cap.set(color='black', linewidth=2)
for median in bp1['medians']:
    median.set(color='black', linewidth=3)
for flier in bp1['fliers']:
    flier.set(marker='D', color='#e7298a', alpha=0.5)

# Overlay mean values for AUROC
for i, data in enumerate(sorted_auroc_data):
    mean_val = np.mean(data)
    ax1.plot(i + 1, mean_val, 'rD', markersize=6)

# Style and font sizes for AUROC
ax1.set_ylabel('AUROC', fontsize=27, fontweight='bold')
plt.xticks(fontsize=27)
ax1.tick_params(axis='y', labelsize=27)
plt.xticks(rotation=45)
plt.tight_layout()

# Save AUROC plot
plt.savefig("AUROC_Boxplot_Pseudotime_lambda1.5.pdf", format='pdf')
plt.close()  # Close the figure to avoid overlap

# Plot AUPRC
fig2, ax2 = plt.subplots(figsize=(12, 8))
bp2 = ax2.boxplot(sorted_auprc_data, patch_artist=True, labels=short_names)
for patch in bp2['boxes']:
    patch.set_facecolor('white')  # Remove color inside boxes
for whisker in bp2['whiskers']:
    whisker.set(color='black', linewidth=1.5)
for cap in bp2['caps']:
    cap.set(color='black', linewidth=2)
for median in bp2['medians']:
    median.set(color='black', linewidth=3)
for flier in bp2['fliers']:
    flier.set(marker='D', color='#e7298a', alpha=0.5)

# Overlay mean values for AUPRC
for i, data in enumerate(sorted_auprc_data):
    mean_val = np.mean(data)
    ax2.plot(i + 1, mean_val, 'rD', markersize=6)

# Style and font sizes for AUPRC
ax2.set_ylabel('AUPRC', fontsize=27, fontweight='bold')
plt.xticks(fontsize=27)
ax2.tick_params(axis='y', labelsize=27)
plt.xticks(rotation=45)
plt.tight_layout()

# Save AUPRC plot
plt.savefig("AUPRC_Boxplot_Pseudotime_lambda1.5.pdf", format='pdf')
plt.close()  # Close the figure
