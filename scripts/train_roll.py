"""Train ONE (net, roll-window) job of the E2E-DRO replication.

Their `net_roll_test` reloads the same init state and re-fits the prediction
layer to OLS at the top of every roll window, so the windows share no state and
can run as independent processes. This script runs exactly one of them and
writes a self-contained chunk; `combine_rolls.py` stitches the chunks back into
a full backtest.

Faithfulness: the training loop below is a line-by-line reimplementation of
`e2edro.e2e_net.net_train` (full-batch gradient accumulation, one Adam step per
epoch, gamma/delta clamped at 1e-4), with per-epoch checkpointing added so a
crash or a killed session loses at most one epoch.

Requires the dedicated env (torch + cvxpy + cvxpylayers + ecos), NOT anaconda base:
    python3 -m venv ~/.venvs/polab && source ~/.venvs/polab/bin/activate
    pip install torch cvxpy cvxpylayers ecos pandas scipy pandas_datareader \\
                alpha_vantage statsmodels psutil

Usage:
    python scripts/train_roll.py --net nom --roll 0
    # all 8 jobs in parallel, 1 torch thread each (the conic solves dominate):
    for n in nom dr; do for r in 0 1 2 3; do
        OMP_NUM_THREADS=1 python scripts/train_roll.py --net $n --roll $r &
    done; done; wait
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# POLAB_VENDOR lets a foreign layout (e.g. Colab, where the upstream repo sits at
# /content/E2E-DRO) point at the clone without mirroring our directory tree.
VENDOR = Path(os.environ.get("POLAB_VENDOR", str(ROOT / "vendor" / "E2E-DRO")))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))

from polab import rolls as R  # noqa: E402

# Net configurations, verbatim from their main.py + the CV winners we recovered
# from their cached objects (cv grid skipped: we pass the winning lr/epochs).
NET_CONFIGS = {
    "base": dict(opt_layer="base_mod", train_pred=True, train_gamma=False,
                 train_delta=False, lr=0.005, epochs=30, pkl="base_net"),
    "nom": dict(opt_layer="nominal", train_pred=True, train_gamma=True,
                train_delta=False, lr=0.02, epochs=50, pkl="nom_net"),
    "dr": dict(opt_layer="hellinger", train_pred=True, train_gamma=True,
               train_delta=True, lr=0.0125, epochs=50, pkl="dr_net"),
    "dr_theta": dict(opt_layer="hellinger", train_pred=True, train_gamma=False,
                     train_delta=False, lr=0.0125, epochs=40,
                     pkl="dr_net_learn_theta"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(NET_CONFIGS), required=True)
    ap.add_argument("--roll", type=int, required=True)
    ap.add_argument("--n-roll", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="override CV winner")
    ap.add_argument("--lr", type=float, default=None, help="override CV winner")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads; keep at 1 when running jobs in parallel")
    ap.add_argument("--out", default=str(ROOT / "results" / "rolls"))
    args = ap.parse_args()

    cfg = NET_CONFIGS[args.net]
    lr = args.lr if args.lr is not None else cfg["lr"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    tag = f"{cfg['pkl']}_roll{args.roll}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = out_dir / f"{tag}.pkl"
    ckpt_path = out_dir / f"{tag}.ckpt"
    if chunk_path.exists():
        print(f"{tag}: already complete ({chunk_path}) — nothing to do")
        return

    os.chdir(VENDOR)  # their relative paths (./cache/...) resolve from here
    import torch
    torch.set_num_threads(max(1, args.threads))

    # PyTorch >=2.6 defaults torch.load to weights_only=True; the only things we
    # load are state dicts written by this process.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from e2edro import e2edro as e2e
    from e2edro import DataLoad as dl
    from e2edro import PortfolioClasses as pc

    # ---- data (their exact call; ships in the vendored cache) ----------------
    init_split = [0.6, 0.4]
    n_obs, perf_period = 104, 13
    X, Y = dl.AV("2000-01-01", "2021-09-30", init_split, freq="weekly",
                 n_obs=n_obs, n_y=20, use_cache=True, save_results=False,
                 AV_key=None)
    n_x, n_y = X.data.shape[1], Y.data.shape[1]

    plan = R.plan(n_total=X.data.shape[0], init_split=tuple(init_split),
                  n_roll=args.n_roll, n_obs=n_obs, perf_period=perf_period)
    win = plan[args.roll]
    print(f"{tag}: split={win.split} train_win={win.n_train_windows} "
          f"test_win={win.n_test_windows} offset={win.offset} "
          f"lr={lr} epochs={epochs}")

    cache_path = str(VENDOR / "new_cache" / "exp") + "/"
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    net = e2e.e2e_net(n_x, n_y, n_obs, prisk="p_var",
                      train_pred=cfg["train_pred"], train_gamma=cfg["train_gamma"],
                      train_delta=cfg["train_delta"], set_seed=1000,
                      opt_layer=cfg["opt_layer"], perf_loss="sharpe_loss",
                      cache_path=cache_path, perf_period=perf_period,
                      pred_loss_factor=0.5).double()

    # ---- roll-window setup (their loop body, verbatim order) ----------------
    X.split_update(list(win.split)), Y.split_update(list(win.split))
    train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), n_obs, perf_period))
    test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), n_obs, 0))
    assert len(train_set) == win.n_train_windows, "train window count mismatch"
    assert len(test_set) == win.n_test_windows, "test window count mismatch"

    net.load_state_dict(torch.load(net.init_state_path))

    # prediction layer initialized to the OLS solution on this window
    X_train, Y_train = X.train(), Y.train()
    X_train.insert(0, "ones", 1.0)
    X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
    Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
    Theta = (torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)).T
    del X_train, Y_train
    with torch.no_grad():
        net.pred_layer.bias.copy_(Theta[:, 0])
        net.pred_layer.weight.copy_(Theta[:, 1:])

    # ---- training (reimplements net_train + checkpointing) ------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    n_train = len(train_set)
    start_epoch = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path)
        net.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        print(f"{tag}: resuming from epoch {start_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = 0.0
        optimizer.zero_grad()
        for x, y, y_perf in train_set:
            z_star, y_hat = net(x.squeeze(), y.squeeze())
            if net.pred_loss is None:
                loss = (1 / n_train) * net.perf_loss(z_star, y_perf.squeeze())
            else:
                loss = (1 / n_train) * (
                    net.perf_loss(z_star, y_perf.squeeze())
                    + (net.pred_loss_factor / net.n_y)
                    * net.pred_loss(y_hat, y_perf.squeeze()[0]))
            loss.backward()
            train_loss += loss.item()
        optimizer.step()
        for name, param in net.named_parameters():
            if name in ("gamma", "delta"):
                param.data.clamp_(0.0001)

        torch.save({"model": net.state_dict(), "optim": optimizer.state_dict(),
                    "epoch": epoch}, ckpt_path)
        done, left = epoch - start_epoch + 1, epochs - epoch - 1
        eta = (time.time() - t_start) / done * left / 60
        print(f"{tag}: epoch {epoch + 1}/{epochs} loss={train_loss:.6f} "
              f"({time.time() - t0:.1f}s, ETA {eta:.0f} min)", flush=True)

    # ---- out-of-sample evaluation of this window ---------------------------
    weights = np.zeros((win.n_test_windows, n_y))
    rets = np.zeros(win.n_test_windows)
    with torch.no_grad():
        for t, (x, y, y_perf) in enumerate(test_set):
            z_star, _ = net(x.squeeze(), y.squeeze())
            weights[t] = z_star.squeeze()
            rets[t] = y_perf.squeeze() @ weights[t]

    chunk = {
        "net": cfg["pkl"], "roll": args.roll, "offset": win.offset,
        "dates": Y.test().index[n_obs:], "weights": weights, "rets": rets,
        "gamma": net.gamma.item() if cfg["train_gamma"] or cfg["opt_layer"] != "base_mod"
                 else None,
        "delta": net.delta.item() if hasattr(net, "delta") else None,
        "lr": lr, "epochs": epochs, "split": win.split,
        "train_seconds": time.time() - t_start,
    }
    with open(chunk_path, "wb") as f:
        pickle.dump(chunk, f)
    ckpt_path.unlink(missing_ok=True)
    print(f"{tag}: DONE -> {chunk_path} "
          f"({chunk['train_seconds'] / 3600:.2f} h, mean ret {rets.mean():.5f})")


if __name__ == "__main__":
    main()
