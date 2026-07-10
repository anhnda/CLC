"""Plot inter-layer mean-shift accumulation curves from a drift_eval JSON.

Usage:
    python scripts/python/plot_drift.py results/eval/awq_drift_b4_drift.json \
        --out results/eval/awq_drift_b4

Produces two panels:
  (a) |mean shift| vs depth  -- E1 accumulated (base vs CLC) + E2 isolated
  (b) relative L2 error vs depth -- E1 accumulated (base vs CLC)

The E1 curves answer the reviewer's question: does the first-moment shift
compound with depth, and does CLC flatten that curve? The E2 overlay shows the
isolated per-layer shift that CLC actually optimizes, so the gap E1 - E2 is the
accumulation component that the current theory does not yet cover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", default=None, help="Output path prefix (no extension)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    depth = list(range(data["num_blocks"]))
    base_e1 = data["base"]["E1_accumulated"]["per_layer_abs_mean_shift"]
    clc_e1 = data["clc"]["E1_accumulated"]["per_layer_abs_mean_shift"]
    base_e2 = data["base"]["E2_isolated"]["per_layer_abs_mean_shift"]
    clc_e2 = data["clc"]["E2_isolated"]["per_layer_abs_mean_shift"]

    base_e1_rel = data["base"]["E1_accumulated"]["per_layer_rel_l2"]
    clc_e1_rel = data["clc"]["E1_accumulated"]["per_layer_rel_l2"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(depth, base_e1, marker="o", label="base (E1 accumulated)")
    ax.plot(depth, clc_e1, marker="o", label="+CLC (E1 accumulated)")
    ax.plot(depth, base_e2, marker="x", linestyle="--", label="base (E2 isolated)")
    ax.plot(depth, clc_e2, marker="x", linestyle="--", label="+CLC (E2 isolated)")
    ax.set_xlabel("decoder block index")
    ax.set_ylabel("|mean output shift|")
    ax.set_title("(a) First-moment shift vs depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(depth, base_e1_rel, marker="o", label="base (E1 accumulated)")
    ax.plot(depth, clc_e1_rel, marker="o", label="+CLC (E1 accumulated)")
    ax.set_xlabel("decoder block index")
    ax.set_ylabel("relative L2 output error")
    ax.set_title("(b) Accumulated relative error vs depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out = args.out or str(Path(args.json_path).with_suffix(""))
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
