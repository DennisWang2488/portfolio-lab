"""Generate the parallel (TPU-runtime) Colab notebook for the E2E-DRO replication.

The worker code is NOT duplicated here: this reads `polab/rolls.py` and
`scripts/train_roll.py` from the repo and inlines them into %%writefile cells,
so the repo stays the single source of truth.

Division of labour:
  Colab  — run the (net, roll) jobs in parallel, produce chunk pickles.
  Local  — `python scripts/combine_rolls.py` stitches and scores them against
           the shipped cache (needs the vendored cache + full polab package).

Usage:  python scripts/make_colab_parallel_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "colab_parallel_retrain.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


def writefile_cell(path_in_colab: str, local_file: Path, note: str) -> dict:
    body = local_file.read_text()
    return code(f"# {note}\n%%writefile {path_in_colab}\n{body}"
                .replace("%%writefile", "%%writefile", 1))


def build() -> dict:
    rolls_src = (ROOT / "polab" / "rolls.py").read_text()
    worker_src = (ROOT / "scripts" / "train_roll.py").read_text()

    cells = [
        md("""# E2E-DRO replication — parallel retraining on a Colab TPU runtime

**为什么用 TPU runtime / Why the TPU runtime**: we do **not** use the TPU itself.
The TPU runtime is selected purely because it ships with many vCPUs, and this
workload is CPU-bound — cvxpylayers/diffcp solves and differentiates thousands of
small conic programs. Do not install or import `torch_xla`; plain CPU torch is
what we want.

**Runtime → Change runtime type → TPU**, then run the cells in order.

**Why this works at all** (the structural finding): `net_roll_test` reloads the
same saved init state and re-fits the prediction layer to OLS at the top of every
roll window, so the 4 windows share no state. They are independent jobs. Total
work is ~146,700 forward+backward passes per net (~16 h serial, which is why the
earlier single-threaded attempt could never fit in a Colab session), but the
windows tile as 114+113+114+113 = 454 OOS weeks and run concurrently — so
wall-clock is the **slowest single roll** (roll 3, ~45,200 steps), not the sum.

**Resilience**: every job checkpoints after each epoch. If the session drops,
re-run the launch cell — finished jobs are skipped and partial ones resume from
their last epoch. Mount Drive (cell 3) to keep checkpoints across sessions."""),

        code("""# 1) Runtime check + dependencies
import os, multiprocessing
print("vCPUs:", multiprocessing.cpu_count())
!nproc; free -g | head -2
# ecos: newer cvxpy no longer bundles it but the layers still request it.
%pip -q install cvxpylayers "cvxpy>=1.4" ecos pandas_datareader alpha_vantage statsmodels psutil
import torch, cvxpy
print("torch", torch.__version__, "| cvxpy", cvxpy.__version__)
assert not torch.cuda.is_available() or True  # GPU irrelevant; CPU path is used"""),

        code("""# 2) Clone the upstream repo (ships the real dataset in cache/*.pkl)
import os, sys, subprocess
REPO = "/content/E2E-DRO"
if not os.path.isdir(REPO):
    !git clone --depth 1 https://github.com/Iyengar-Lab/E2E-DRO.git {REPO}
os.makedirs(REPO + "/new_cache/exp", exist_ok=True)
assert os.path.isdir(REPO + "/e2edro") and os.path.isdir(REPO + "/cache")
print("repo OK:", REPO)
!ls {REPO}/cache/"""),

        code('''# 3) OPTIONAL but recommended — persist chunks/checkpoints on Drive so a
# disconnect does not lose hours of training. Skip if you prefer ephemeral runs.
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    CHUNKS = '/content/drive/MyDrive/polab_rolls'
else:
    CHUNKS = '/content/polab_rolls'
LOGS = '/content/polab_logs'
os.makedirs(CHUNKS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
print("chunks ->", CHUNKS)
print("logs   ->", LOGS)'''),

        code("# 4) Write the polab.rolls module (split arithmetic; verified offline\n"
             "#    against their 454 OOS weeks)\n"
             "import os\n"
             "os.makedirs('/content/polab', exist_ok=True)\n"
             "open('/content/polab/__init__.py','w').write('')\n"
             "_src = r'''" + rolls_src + "'''\n"
             "open('/content/polab/rolls.py','w').write(_src)\n"
             "import sys; sys.path.insert(0, '/content')\n"
             "from polab import rolls as R\n"
             "print(R.summary(R.plan()))"),

        code("# 5) Write the worker script (one (net, roll) job, resumable)\n"
             "os.makedirs('/content/scripts', exist_ok=True)\n"
             "_src = r'''" + worker_src + "'''\n"
             "open('/content/scripts/train_roll.py','w').write(_src)\n"
             "print('worker written:', len(_src), 'bytes')"),

        code('''# 6) DIAGNOSTIC — 1 forward+backward, ~1 min. Confirms the training path
# works and gives a real per-step time so the ETA below is grounded.
import os, sys, time, traceback, torch
os.chdir(REPO); sys.path.insert(0, REPO)
torch.set_num_threads(1)
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

from torch.utils.data import DataLoader
from e2edro import e2edro as e2e, DataLoad as dl, PortfolioClasses as pc

X, Y = dl.AV("2000-01-01", "2021-09-30", [0.6, 0.4], freq="weekly",
             n_obs=104, n_y=20, use_cache=True, save_results=False, AV_key=None)
n_x, n_y = X.data.shape[1], Y.data.shape[1]
print("data:", X.data.shape, Y.data.shape)

diag = e2e.e2e_net(n_x, n_y, 104, prisk="p_var", train_pred=True, train_gamma=True,
                   train_delta=False, set_seed=1000, opt_layer="nominal",
                   perf_loss="sharpe_loss", cache_path=REPO + "/new_cache/exp/",
                   perf_period=13, pred_loss_factor=0.5).double()
diag.load_state_dict(torch.load(diag.init_state_path))
print("init state reload: OK")

X.split_update([0.6, 0.1]); Y.split_update([0.6, 0.1])
ts = DataLoader(pc.SlidingWindow(X.train(), Y.train(), 104, 13))
x, y, y_perf = next(iter(ts))
try:
    t0 = time.time(); z, yh = diag(x.squeeze(), y.squeeze()); tf = time.time() - t0
    loss = diag.perf_loss(z, y_perf.squeeze())
    t0 = time.time(); loss.backward(); tb = time.time() - t0
    print(f"forward {tf:.2f}s + backward {tb:.2f}s = {tf+tb:.2f}s/step")
    print(f"slowest roll (roll 3, 45,200 steps) ETA: "
          f"{45200*(tf+tb)/3600:.1f} h  <-- this is the wall-clock to expect")
except Exception:
    traceback.print_exc()
X.split_update([0.6, 0.4]); Y.split_update([0.6, 0.4])'''),

        code('''# 7) LAUNCH — all (net, roll) jobs in parallel, one process each.
# Each job is pinned to 1 thread: the conic solves are single-threaded, so N
# processes on N cores scale nearly linearly. Re-run this cell after a
# disconnect — completed jobs are skipped, partial ones resume.
import subprocess, sys, os

NETS = ["nom", "dr", "dr_theta"]     # the headline comparison + the cost-stress winner
ROLLS = [0, 1, 2, 3]

env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "POLAB_VENDOR": REPO}
procs = {}
for net in NETS:
    for r in ROLLS:
        tag = f"{net}_roll{r}"
        # No skip logic needed: the worker exits immediately if its chunk
        # already exists, and resumes from the last epoch if a .ckpt is there.
        log = open(f"{LOGS}/{tag}.log", "a")
        procs[tag] = subprocess.Popen(
            [sys.executable, "/content/scripts/train_roll.py",
             "--net", net, "--roll", str(r), "--out", CHUNKS, "--threads", "1"],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd="/content")
print(f"launched {len(procs)} jobs on {os.cpu_count()} vCPUs "
      f"(already-complete ones will exit within seconds)")'''),

        code('''# 8) MONITOR — re-runnable. Keep this cell running to hold the session open.
import time, glob, os
while True:
    alive = {t: p for t, p in procs.items() if p.poll() is None}
    done = sorted(os.path.basename(f) for f in glob.glob(f"{CHUNKS}/*.pkl"))
    print(f"[{time.strftime('%H:%M:%S')}] running={len(alive)} chunks={len(done)}")
    for tag in sorted(procs):
        log = f"{LOGS}/{tag}.log"
        last = ""
        if os.path.exists(log):
            lines = [l.strip() for l in open(log).readlines() if l.strip()]
            last = lines[-1][-90:] if lines else ""
        state = "RUN " if procs[tag].poll() is None else f"EXIT{procs[tag].poll()}"
        print(f"  {state} {tag:22s} {last}")
    if not alive:
        print("all jobs finished"); break
    time.sleep(120)'''),

        code('''# 9) Package the chunks for download (combine + scoring happens LOCALLY,
# where the vendored cache and the full polab package live)
import shutil
shutil.make_archive('/content/polab_chunks', 'zip', CHUNKS)
!ls -la /content/polab_chunks.zip
from google.colab import files
files.download('/content/polab_chunks.zip')'''),

        md("""## 带回本地 / Bring the results home

```bash
unzip -o ~/Downloads/polab_chunks.zip -d research-projects/portfolio-lab/results/rolls/
python scripts/combine_rolls.py
```

`combine_rolls.py` stitches the four windows per net (verifying the assembled
length is exactly 454), then reports retrained vs shipped-cache Sharpe.

**Success criterion**: the ranking `dr_net > nom_net > 1/N > po_net` is preserved
and the Sharpes land near the cached reference (dr 1.314, nom 1.178,
dr_learn_theta 1.414). Exact reproduction is *not* expected — torch/cvxpy/
cvxpylayers have all moved several versions since 2022, even with their seed
(1000). A large deviation is a **finding**, not a failure; record it either way
in `notes.md`.

**If the session drops**: reconnect, re-run cells 1–5 (fast), then cell 7. With
Drive mounted, finished jobs are skipped and partial ones resume from their last
checkpointed epoch."""),
    ]

    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": []},
                         "accelerator": "TPU"},
            "cells": cells}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")
