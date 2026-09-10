"""
Task 5 validation gate (pseudogrn_fix_and_reanalyze_prompt.md): explicitly
read back n_cells_pooled_full / n_cells_written (from prep_meta.json) and
n_cells_pooled (from metrics.json) for the 3 corrected single-seed runs,
and print a direct comparison -- confirming they are genuinely DIFFERENT
and correctly ORDERED (Tier1(capped) > Tier2(natural,13500) >
Tier3(natural,8100)), not just trusting the code path. This is exactly
the check that would have caught the original --max_cells=3000 bug (every
tier silently capped to the same value) before it reached a full 9-run
sweep.

Run AFTER the 3 (tier, seed=0) combos have completed.

USAGE
-----
python check_cell_counts.py --data_root . --results_root PseudoGRN_results/SERGIO
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--results_root", type=str, default="PseudoGRN_results/SERGIO")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)

    rows = []
    for tier in [1, 2, 3]:
        runid = f"tier{tier}_seed{args.seed}"
        prep_meta_path = data_root / f"data_{runid}" / "prep_meta.json"
        metrics_path = results_root / runid / "metrics.json"

        missing = [p for p in (prep_meta_path, metrics_path) if not p.exists()]
        if missing:
            print(f"MISSING for tier{tier}: {[str(p) for p in missing]} -- run the sweep first")
            continue

        prep = json.load(open(prep_meta_path))
        metrics = json.load(open(metrics_path))
        rows.append(dict(
            tier=tier,
            n_cells_pooled_full=prep["n_cells_pooled_full"],
            n_cells_written_prep=prep["n_cells_written"],
            max_cells_cap=prep["max_cells_cap"],
            n_cells_pooled_metrics=metrics["n_cells_pooled"],
            mean_auprc=metrics["mean_auprc"],
            mean_auroc=metrics["mean_auroc"],
        ))

    if len(rows) < 3:
        print(f"\nOnly {len(rows)}/3 runs found -- cannot complete the validation gate yet.")
        return

    print(f"{'tier':<6}{'natural_pool':>14}{'written(prep)':>16}{'written(metrics)':>18}{'cap':>8}{'AUPRC':>10}{'AUROC':>10}")
    for r in rows:
        print(f"{r['tier']:<6}{r['n_cells_pooled_full']:>14}{r['n_cells_written_prep']:>16}"
              f"{r['n_cells_pooled_metrics']:>18}{r['max_cells_cap']:>8}"
              f"{r['mean_auprc']:>10.4f}{r['mean_auroc']:>10.4f}")

    written = [r["n_cells_written_prep"] for r in rows]
    print()

    if len(set(written)) == 3:
        print(f"PASS: all 3 tiers have DISTINCT n_cells_written: {written}")
    else:
        print(f"FAIL: n_cells_written values are NOT all distinct: {written} "
              f"-- the confound may still be present!")

    if written[0] > written[1] > written[2]:
        print(f"PASS: correctly ordered Tier1({written[0]}) > Tier2({written[1]}) > Tier3({written[2]})")
    else:
        print(f"FAIL: expected Tier1 > Tier2 > Tier3, got {written}")

    for r in rows:
        if r["n_cells_written_prep"] != r["n_cells_pooled_metrics"]:
            print(f"WARNING: tier{r['tier']}: prep_meta n_cells_written="
                  f"{r['n_cells_written_prep']} != metrics.json n_cells_pooled="
                  f"{r['n_cells_pooled_metrics']} -- unexpected mismatch between "
                  f"what was prepared and what was actually trained on")

    expected_natural = {1: 40500, 2: 13500, 3: 8100}
    for r in rows:
        if r["n_cells_pooled_full"] != expected_natural[r["tier"]]:
            print(f"NOTE: tier{r['tier']} natural pool ({r['n_cells_pooled_full']}) "
                  f"differs from the expected {expected_natural[r['tier']]} -- "
                  f"double check --n_replicates/--n_bins match every other model's "
                  f"convention if this wasn't intentional.")


if __name__ == "__main__":
    main()
