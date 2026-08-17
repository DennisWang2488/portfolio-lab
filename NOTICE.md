# Third-party code and data

Nothing in this list is redistributed in this repository. Each item is fetched
by a setup script at its own source, under its own license. This file records
what the lab depends on and who wrote it.

## Code

**E2E-DRO** — Giorgio Costa and Garud N. Iyengar, *Distributionally Robust
End-to-End Portfolio Construction* (Quantitative Finance, 2023).
Columbia University Iyengar Lab. Licensed **Apache License 2.0**.

- Upstream: <https://github.com/Iyengar-Lab/E2E-DRO>
- Fetched by: `bash scripts/setup_vendor.sh` → `vendor/E2E-DRO/`
- Used for: their weekly returns panel (20 US large-caps + 8 Fama–French
  factors, 2000–2021) and their cached trained networks, so that the
  replication in `scripts/compare_e2edro_cache.py` runs against *their* numbers
  rather than a re-implementation of them.
- Modifications: none to their source. `polab/e2edro_io.py` unpickles their
  cached objects in an environment without their optional dependencies by
  stubbing the missing modules; this is done at import time in our code and
  leaves the vendored tree untouched.

## Data

**Daily sector-ETF prices** — nine SPDR select-sector ETFs (XLB XLE XLF XLI XLK
XLP XLU XLV XLY) plus SPY, adjusted closes and volume, 2014–2024, from the
Yahoo Finance chart endpoint.

- Fetched by: `python scripts/fetch_etf_data.py` → `data/yahoo_daily/`
- Not redistributed: Yahoo's terms of service do not permit republishing their
  price series. Re-fetching gives a possibly different data *vintage* (Yahoo
  restates adjusted closes after corporate actions), which is why the fetch
  script prints the window and row count it obtained.

**Fama–French factors** reach this repository only through the E2E-DRO cache
above. The originals are Kenneth R. French's, distributed free for research at
<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>.

## Papers replicated or audited

- Costa & Iyengar (2023), as above — replicated in `scripts/compare_e2edro_cache.py`.
- arXiv:2601.04062 (January 2026), which claims decision-focused (SPO+) training
  consistently beats predict-then-optimize on US ETF data and released no code.
  Audited in [`audits/spo-2601.04062.md`](audits/spo-2601.04062.md); independently
  re-tested under the pre-registration in
  [`audits/prereg-spo-retest.md`](audits/prereg-spo-retest.md). The re-test uses
  **our** universe, not theirs — theirs is not recoverable from the paper — and
  no result here is described as reproducing their numbers.
