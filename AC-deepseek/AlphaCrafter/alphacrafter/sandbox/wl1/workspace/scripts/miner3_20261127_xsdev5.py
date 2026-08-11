"""miner_3 candidate: xs_dev_5 (cross-sectional deviation of 5d return).

Motivation: cycle-8 screen flagged xs_dev_5 as the strongest low-rho candidate
(ic=-0.065, icir=-0.180 on window through 2026-08). Re-validate on the extended
window through the last completed trading day (2026-11).

Construction: for each date, subtract the cross-sectional mean of the 5-day
return from each asset's own 5-day return. Assets that deviated above the
cross-section tend to revert (negative IC). Rank IC is invariant to the
cross-sectional demeaning, so this is the raw 5d relative-strength deviation.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel, run_validation, make_artifact

close = load_close_panel()
ret5 = close.pct_change(5)
xs_mean5 = ret5.mean(axis=1)
factor = ret5.sub(xs_mean5, axis=0)

# report construction sanity
print(f"panel dates={close.shape[0]} assets={close.shape[1]} "
      f"range={close.index.min().date()}..{close.index.max().date()}")

notes = ("Cross-sectional deviation of 5d return (demeaned relative strength). "
         "Validated on 15-asset tradable universe through 2026-11-26. "
         "Regimes: 2020 COVID crash, 2021 bull, 2022 bear, 2023-24 AI rally, "
         "2025-26 crypto/commodity cycles, 2026 rate regime. "
         "Direction: negative IC -> short relative winners / long relative losers.")
summary = run_validation(factor, close, factor_id="miner3_20261127_xs_dev_5",
                         regime_notes=notes, return_summary=True)
if summary:
    summary["direction"] = -1
    summary["artifact"] = make_artifact(factor)
    print("SUMMARY_JSON", json_dumps := __import__("json").dumps(summary)[:200])
