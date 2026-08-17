"""Insert the training-failure diagnostic cells into the Colab notebook.

Idempotent: re-running replaces the diagnostic cells rather than duplicating
them. Kept in the repo (not the scratchpad) so it survives session cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "colab_retrain_e2edro.ipynb"

DIAGNOSTIC = '''# 6.5) DIAGNOSTIC — isolate any training failure in <1 min, and project runtime.
# Run this BEFORE committing hours to cells 7/8. The smoke tests above exercise
# the CvxpyLayer FORWARD path only; training adds two untested things: reloading
# the init state, and backprop through the optimization layer.
cd_repo()
import time, traceback, torch
from torch.utils.data import DataLoader
from e2edro import PortfolioClasses as pc

diag = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                   train_pred=True, train_gamma=True, train_delta=False,
                   set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss,
                   cache_path=cache_path, perf_period=perf_period,
                   pred_loss_factor=pred_loss_factor).double()
print("A) construct + save init state: OK")

# B) net_roll_test reloads the init state at the top of every roll window.
#    PyTorch >= 2.6 flipped torch.load's weights_only default to True.
try:
    diag.load_state_dict(torch.load(diag.init_state_path))
    print("B) torch.load(init_state): OK (no patch needed)")
except Exception:
    traceback.print_exc()
    diag.load_state_dict(torch.load(diag.init_state_path, weights_only=False))
    print("B) torch.load FAILED with the new default; weights_only=False WORKS "
          "-> run cell 6.6 to patch, then rerun this cell")

# C) one forward + backward through the optimization layer (the untested path)
X.split_update([0.6, 0.1]); Y.split_update([0.6, 0.1])
train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), n_obs, perf_period))
n_win = len(train_set)
x, y, y_perf = next(iter(train_set))
try:
    t0 = time.time(); z_star, y_hat = diag(x.squeeze(), y.squeeze()); t_f = time.time() - t0
    loss = diag.perf_loss(z_star, y_perf.squeeze())
    t0 = time.time(); loss.backward(); t_b = time.time() - t0
    print(f"C) forward {t_f:.2f}s + backward {t_b:.2f}s: OK")
    # 50 epochs x windows x 4 roll windows (train set grows ~15% on average)
    steps = 50 * n_win * 4 * 1.15
    print(f"   roll-1 train windows: {n_win} | projected nom_net total: "
          f"{steps * (t_f + t_b) / 3600:.1f} h  <-- CHECK BEFORE PROCEEDING")
except Exception:
    print("C) forward/backward FAILED:")
    traceback.print_exc()

X.split_update(split); Y.split_update(split)   # restore the original split'''

PATCH = '''# 6.6) PATCH — only run if cell 6.5 reported B) FAILED.
# Restore torch.load's pre-2.6 behaviour for this session. Safe here: the only
# things we load are state dicts this notebook itself just wrote.
import torch
if not getattr(torch, "_e2edro_patched", False):
    _orig_load = torch.load
    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _load
    torch._e2edro_patched = True
    print("torch.load patched -> weights_only=False by default")
else:
    print("already patched")'''


def main() -> None:
    nb = json.loads(NB.read_text())
    cells = [c for c in nb["cells"]
             if not c["source"].lstrip().startswith(("# 6.5)", "# 6.6)"))]

    # insert right after the po_net smoke test (cell "# 6)")
    idx = next(i for i, c in enumerate(cells)
               if c["source"].lstrip().startswith("# 6)")) + 1
    new = [{"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}
           for src in (DIAGNOSTIC, PATCH)]
    cells[idx:idx] = new
    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"patched {NB} -> {len(cells)} cells")


if __name__ == "__main__":
    main()
