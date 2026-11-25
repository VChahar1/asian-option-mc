from __future__ import annotations

import numpy as np
from scipy.stats import norm


def geometric_asian_call_closed_form(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
) -> float:
    """Price a discretely sampled geometric-average call under GBM."""
    sigma_g = sigma * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
    mu_g = (r - 0.5 * sigma**2) * (n_steps + 1) / (2 * n_steps) + 0.5 * sigma_g**2

    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    return float(
        np.exp(-r * T) * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
    )


def simulate_average_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate arithmetic and geometric averages with antithetic GBM paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    half = n_paths // 2
    Z = rng.standard_normal((half, n_steps))
    Z = np.concatenate([Z, -Z], axis=0)

    log_increments = drift + diffusion * Z
    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
    paths = np.exp(log_paths)

    arith = paths.mean(axis=1)
    geo = np.exp(np.log(paths).mean(axis=1))
    return arith, geo


def price_arithmetic_asian(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 50,
    n_paths: int = 100_000,
    seed: int | None = None,
    use_control_variate: bool = True,
) -> dict:
    """Estimate an arithmetic-average call, optionally using a geometric control."""
    arith_avg, geo_avg = simulate_average_paths(S0, T, r, sigma, n_steps, n_paths, seed)
    disc = np.exp(-r * T)

    arith_payoff = disc * np.maximum(arith_avg - K, 0.0)
    plain_price = arith_payoff.mean()
    plain_se = arith_payoff.std() / np.sqrt(n_paths)

    if not use_control_variate:
        return {
            "price": float(plain_price),
            "stderr": float(plain_se),
            "method": "plain_mc",
        }

    geo_payoff = disc * np.maximum(geo_avg - K, 0.0)
    geo_true = geometric_asian_call_closed_form(S0, K, T, r, sigma, n_steps)

    cov = np.cov(arith_payoff, geo_payoff)[0, 1]
    var_y = geo_payoff.var()
    c = cov / var_y if var_y > 0 else 0.0

    cv_estimator = arith_payoff - c * (geo_payoff - geo_true)
    cv_price = cv_estimator.mean()
    cv_se = cv_estimator.std() / np.sqrt(n_paths)

    variance_reduction = (plain_se / cv_se) ** 2 if cv_se > 0 else float("inf")

    return {
        "price": float(cv_price),
        "stderr": float(cv_se),
        "plain_price": float(plain_price),
        "plain_stderr": float(plain_se),
        "control_coefficient": float(c),
        "variance_reduction_factor": float(variance_reduction),
        "method": "control_variate",
    }


def _demo() -> None:
    """Demonstration run, printed to stdout."""
    params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=50)

    plain = price_arithmetic_asian(
        **params, n_paths=100_000, seed=0, use_control_variate=False
    )
    cv = price_arithmetic_asian(
        **params, n_paths=100_000, seed=0, use_control_variate=True
    )

    print("Arithmetic Asian call pricing (S0=100, K=100, T=1, r=0.05, sigma=0.2)")
    print(f"  Plain MC:          {plain['price']:.4f} {plain['stderr']:.4f}")
    print(f"  Control variate:   {cv['price']:.4f} {cv['stderr']:.4f}")
    print(f"  Variance reduction: {cv['variance_reduction_factor']:.1f}")
    print(f"  Control coefficient: {cv['control_coefficient']:.4f}")


if __name__ == "__main__":
    _demo()
