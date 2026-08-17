"""Load Costa & Iyengar's cached E2E-DRO experiment objects without their env.

Their pickles reference cvxpylayers / alpha_vantage / pandas_datareader /
statsmodels, none of which we need for reading results — unpickling only needs
importable class definitions, so `install_stubs()` seeds sys.modules with
attribute-friendly stand-ins for whatever is missing. Loading goes through
pd.read_pickle to get pandas' pickle-compat shims (their pickles were written
by an older pandas).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "E2E-DRO"
CACHE = VENDOR / "cache" / "exp"          # their original shipped artifact
NEW_CACHE = VENDOR / "new_cache" / "exp"  # our Colab-retrained pickles land here

# Main-experiment nets (their main.py, experiments 1-4).
MAIN_NETS = {
    "ew_net": "their equal weight",
    "po_net": "predict-then-optimize",
    "base_net": "naive E2E",
    "nom_net": "nominal E2E",
    "dr_net": "DR E2E (Hellinger)",
    "dr_net_learn_delta": "DR E2E, learn delta",
    "dr_net_learn_gamma": "DR E2E, learn gamma",
    "dr_net_learn_theta": "DR E2E, learn theta",
}


def _dummy(name: str) -> type:
    return type(name, (), {})


def install_stubs() -> None:
    """Make e2edro importable/unpicklable in an env without its optional deps."""
    stubs = [
        ("cvxpylayers", {}),
        ("cvxpylayers.torch", {"CvxpyLayer": _dummy("CvxpyLayer")}),
        ("cvxpylayers.torch.cvxpylayer", {"CvxpyLayer": _dummy("CvxpyLayer")}),
        ("pandas_datareader", {"DataReader": _dummy("DataReader")}),
        ("alpha_vantage", {}),
        ("alpha_vantage.timeseries", {"TimeSeries": _dummy("TimeSeries")}),
        ("statsmodels", {}),
        ("statsmodels.api", {"OLS": _dummy("OLS")}),
    ]
    for name, attrs in stubs:
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            for a, v in attrs.items():
                setattr(mod, a, v)
            sys.modules[name] = mod

    import psutil
    if psutil.cpu_count() is None:  # sandboxed environments can return None
        psutil.cpu_count = lambda *a, **k: 4

    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def load_net(name: str, cache_dir: Path = CACHE):
    """Unpickle one cached net object (e.g. 'dr_net'). Raises on failure."""
    install_stubs()
    path = cache_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path)


def net_returns(net) -> pd.Series:
    """Gross weekly OOS portfolio returns of a cached net."""
    rets = net.portfolio.rets  # DataFrame(rets, tri), index Date
    return pd.Series(rets["rets"].to_numpy(), index=pd.DatetimeIndex(rets.index))


def net_weights(net) -> pd.DataFrame:
    """Weights in force at the start of each OOS week (rows align with returns)."""
    w = np.asarray(net.portfolio.weights)
    idx = pd.DatetimeIndex(net.portfolio.rets.index)
    return pd.DataFrame(w, index=idx)
