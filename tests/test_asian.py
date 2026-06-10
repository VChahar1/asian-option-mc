"""Tests for the Asian option pricer."""
from __future__ import annotations

import numpy as np
import pytest

from src.asian import (
    geometric_asian_call_closed_form,
    price_arithmetic_asian,
    simulate_average_paths,
)


def test_geometric_closed_form_positive():
    price = geometric_asian_call_closed_form(100, 100, 1.0, 0.05, 0.2, 50)
    print(price)
    assert 0 < price < 20


def test_mc_geometric_matches_closed_form():
    S0, K, T, r, sigma, n_steps = 100, 100, 1.0, 0.05, 0.2, 50
    arith, geo = simulate_average_paths(S0, T, r, sigma, n_steps,
                                        n_paths=200_000, seed=1)
    disc = np.exp(-r * T)
    geo_mc = (disc * np.maximum(geo - K, 0.0)).mean()
    geo_cf = geometric_asian_call_closed_form(S0, K, T, r, sigma, n_steps)
    geo_se = (disc * np.maximum(geo - K, 0.0)).std() / np.sqrt(200_000)
    assert abs(geo_mc - geo_cf) < 4 * geo_se


def test_control_variate_reduces_variance():
    params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=50)
    plain = price_arithmetic_asian(**params, n_paths=50_000, seed=2,
                                   use_control_variate=False)
    cv = price_arithmetic_asian(**params, n_paths=50_000, seed=2,
                                use_control_variate=True)
    assert cv["stderr"] < plain["stderr"]
    assert cv["variance_reduction_factor"] > 2.0


def test_arithmetic_price_near_geometric():
    params = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=50)
    arith = price_arithmetic_asian(**params, n_paths=200_000, seed=3)
    geo_cf = geometric_asian_call_closed_form(100, 100, 1.0, 0.05, 0.2, 50)
    # Arithmetic >= geometric by AM-GM; difference is small.
    assert arith["price"] >= geo_cf - 0.05
    assert arith["price"] - geo_cf < 1.0