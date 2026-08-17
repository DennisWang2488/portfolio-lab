#!/usr/bin/env bash
# Fetch the third-party code this lab replicates.
#
# Costa & Iyengar (2023), "Distributionally Robust End-to-End Portfolio
# Construction" (Apache 2.0). Their repository ships both the source and the
# cached experiment artifacts — the weekly returns panel and the trained nets —
# which is what makes an apples-to-apples replication possible.
#
# It is NOT redistributed here: it is their work under their license, ~70 MB of
# it, and vendoring someone else's dataset into a portfolio repo is the wrong
# default. Run this once; everything afterwards is offline.
#
#   bash scripts/setup_vendor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
TARGET="$VENDOR/E2E-DRO"
UPSTREAM="https://github.com/Iyengar-Lab/E2E-DRO.git"

if [ -d "$TARGET/.git" ]; then
  echo "already present: $TARGET"
else
  mkdir -p "$VENDOR"
  echo "cloning $UPSTREAM -> $TARGET"
  git clone --depth 1 "$UPSTREAM" "$TARGET"
fi

missing=0
for f in cache/asset_weekly.pkl cache/factor_weekly.pkl; do
  if [ ! -f "$TARGET/$f" ]; then
    echo "WARNING: expected $f is not in the clone." >&2
    missing=1
  fi
done

if [ "$missing" -eq 1 ]; then
  cat >&2 <<'EOF'

The upstream repository may have reorganized its cached artifacts since this
was written. polab/data.py expects:

    vendor/E2E-DRO/cache/asset_weekly.pkl    20 US large-caps, weekly, 2000-2021
    vendor/E2E-DRO/cache/factor_weekly.pkl   8 Fama-French factors, same index

and polab/e2edro_io.py expects their trained nets under:

    vendor/E2E-DRO/cache/exp/*.pkl

Scripts that need only the ETF panel (run_spo_retest.py) or nothing at all
(run_ic_breadth_stability.py, tests/) are unaffected.
EOF
  exit 1
fi

echo
echo "vendor ready. Next:"
echo "  python scripts/fetch_etf_data.py     # daily sector-ETF panel (Yahoo)"
echo "  python tests/test_polab.py           # offline suite, no data needed"
