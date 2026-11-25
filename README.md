# asian-option-mc

Pricing arithmetic-average Asian options by Monte Carlo, using the geometric-average Asian option as a control variate for variance reduction.


## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.asian          # prints the pricing comparison
python -m src.make_figure    # regenerates the convergence figure
pytest -v                    # 4 tests
```

## Verification

The implementation is checked against known properties:
- The Monte Carlo geometric-Asian estimate matches its closed-form value within Monte Carlo error.
- Arithmetic ≥ geometric price (AM-GM inequality on the averages).
- The control-variate estimator has strictly lower variance than plain MC.

## Notes and extensions

This module prices fixed-strike arithmetic Asian calls. Natural extensions: put options (trivial), floating-strike Asians, control variates using the underlying itself rather than the geometric Asian, and quasi-Monte Carlo (Sobol sequences) for further variance reduction. The geometric control variate is the highest-leverage single technique and is standard in production Asian-option pricing.