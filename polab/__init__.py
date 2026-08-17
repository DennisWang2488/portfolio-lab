"""polab — honest portfolio-optimization lab.

Baselines + strictly causal walk-forward backtesting + multiple-testing-aware
metrics. See README.md for the project charter.
"""

from . import backtest, baselines, data, metrics, sharpe_test

__all__ = ["backtest", "baselines", "data", "metrics", "sharpe_test"]
__version__ = "0.1.0"
