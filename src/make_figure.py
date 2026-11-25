from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.asian import price_arithmetic_asian


def main() -> None:
    """Plot the model-reported standard errors from seeded simulations."""
    params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=50)
    path_counts = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    plain_se, cv_se = [], []
    for n in path_counts:
        p = price_arithmetic_asian(
            **params, n_paths=n, seed=0, use_control_variate=False
        )
        c = price_arithmetic_asian(
            **params, n_paths=n, seed=0, use_control_variate=True
        )
        plain_se.append(p["stderr"])
        cv_se.append(c["stderr"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(path_counts, plain_se, "o-", label="Plain Monte Carlo")
    ax.loglog(path_counts, cv_se, "s-", label="With geometric control variate")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Standard error")
    ax.set_title("Reported Monte Carlo error with a geometric control")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/results.png", dpi=120)
    print("Saved assets/results.png")


if __name__ == "__main__":
    main()
