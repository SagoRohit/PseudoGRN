# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


def concat_ref(df, df_r):
    '''
    Concat outputFile and referenceNetwork to get the ground truth of the edges

    Input: 
    df: dataFrame of the outputFile
    df_r: dataFrame of the reference network

    Output: dataFrame of the outputFile with ground truth
    '''
    df.columns = ['Gene1', 'Gene2', 'score']
    #df_r.columns = ['Gene1', 'Gene2']
    df_r['real_score'] = 1
    df = pd.merge(df, df_r, how='left')
    df.loc[df.real_score.isna(), 'real_score'] = 0
    df = df.loc[df.Gene1!=df.Gene2].drop_duplicates(keep='first')
    return df


def cal_auc_aupr(df):
    '''
    Evaluate auroc and auprc of the algorithm on a specific dataset

    Input: 
    df: dataFrame of the outputFile with ground truth

    Output: a dict with keys ['auroc', 'auprc']
    '''
    scores = df.score.values
    true = df.real_score.astype(int).values
    fpr, tpr, thresholds = metrics.roc_curve(true, scores, pos_label=1)
    prec, recall, thresholds = metrics.precision_recall_curve(true, scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(recall, prec)
    return {'AUROC':auroc, 'AUPRC':auprc}


def cal_EPR(df, k, sparsity):
    '''
    Evaluate EP, EPR of the algorithm on a specific dataset

    Input: 
    df: dataFrame of the inferred network with ground truth
    k: the top k edges are taken for evaluation 
    sparsity: the sparsity of the real network

    Output: a dict with keys of EP, EPR on the top k edges
    '''
    k = min(k, len(df))
    df = df.sort_values(by='score', ascending=False)
    TP = df.iloc[:k, :].real_score.sum().astype(int)
    EP = TP/k
    EPR = EP/sparsity
    return TP, EP, EPR


########################### partial corr to decide activation/inhibition
def add_sign_and_plot(df_edge, expression_file, top_k=None, plot=True, output_pdf=None):
    """
    Plot gene regulatory network:
    - Black solid arrow: Activation (positive partial correlation)
    - Red dashed arrow: Inhibition (negative partial correlation)
    Arrow direction follows df_edge (e.g., from KL or JS scores)
    """
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from sklearn.covariance import GraphicalLassoCV
    from matplotlib.patches import FancyArrowPatch

    # Load expression data
    df_exp = pd.read_csv(expression_file, index_col=0)
    all_genes = sorted(df_exp.columns.tolist())

    # Limit to top_k edges
    df_edge = df_edge.sort_values('score', ascending=False)
    if top_k is not None:
        df_edge = df_edge.head(top_k)

    # Drop duplicate directed edges
    df_edge = df_edge.drop_duplicates(subset=['Gene1', 'Gene2'])

    # Compute partial correlation matrix
    X = df_exp.values
    model = GraphicalLassoCV().fit(X)
    precision = model.precision_
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    gene_to_idx = {g: i for i, g in enumerate(df_exp.columns)}

    # Create graph and layout
    G = nx.DiGraph()
    G.add_nodes_from(all_genes)
    pos = nx.circular_layout(G)

    # Start plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw directed edges with arrows and color based on sign
    for _, row in df_edge.iterrows():
        g1, g2 = row['Gene1'], row['Gene2']
        if g1 not in gene_to_idx or g2 not in gene_to_idx:
            continue
        val = partial_corr[gene_to_idx[g1], gene_to_idx[g2]]
        if np.isnan(val):
            continue

        x_start, y_start = pos[g1]
        x_end, y_end = pos[g2]

        # Assign style
        color = 'black' if val > 0 else 'red'
        linestyle = 'solid' if val > 0 else (0, (5, 5))  # dashed

        arrow = FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            connectionstyle="arc3,rad=0.1",
            arrowstyle='-|>',
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            mutation_scale=25,   # arrow size
            shrinkA=12, shrinkB=12,
            zorder=1
        )
        ax.add_patch(arrow)

    # Add custom legend
    from matplotlib.lines import Line2D

    # Add custom legend using Line2D instead of FancyArrowPatch
    legend_handles = [
    Line2D([0], [0], color='black', linewidth=0.7, linestyle='solid', label='Activation'),
    Line2D([0], [0], color='red', linewidth=0.7, linestyle=(0, (5, 3)), label='Inhibition')
]
    plt.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=10)
    plt.title("Gene Regulatory Network (19 Genes)", fontsize=14)
    plt.axis('off')

    if output_pdf:
        plt.savefig(output_pdf, bbox_inches='tight', dpi=300, format='pdf')
    if plot:
        plt.show()

    return df_edge
