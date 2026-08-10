"""miner_1 2026-07-30 cycle 23: carry / term-structure family exploration.

Idea: cross-asset "carry" (Koijen-Moskowitz-Pedersen-Vrugt style) approximated
from the spot price curve -- carry = return earned if prices stay flat. With only
daily OHLC we proxy carry by the term-structure of trailing returns:
    carry(a,b) = return over (a..b ago)  -  return over (0..a ago)
i.e. how much of the longer-window return was earned in the EARLIER part vs the
MOST RECENT part. Positive => backwardation-like (roll yield positive) profile.

Variants: 12m-1m, 6m-1m, 3m-1m, 12m-3m. All computed per-asset on own calendar
(crypto/equity calendars differ), reindexed to union panel. Validation uses the
same harness as the library (admission h=10, |IC|>=0.007, |ICIR|>=0.084).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, load_library_signals, report)

panel = load_panel()
print(f"panel shape: {panel.shape}  dates: {panel.index.min().date()}..{panel.index.max().date()}")

HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------------------------------------------------------------------
# Carry / term-structure constructions (per-asset own-calendar)
# ---------------------------------------------------------------------------
CAND = {
    "carry_12m1m": ("lambda s: (s.shift(21)/s.shift(252)-1.0) - (s/s.shift(21)-1.0)",
                    "12m-1m carry proxy: 12m return (ex last month) minus last-month return"),
    "carry_6m1m":  ("lambda s: (s.shift(21)/s.shift(126)-1.0) - (s/s.shift(21)-1.0)",
                    "6m-1m carry proxy: 6m return (ex last month) minus last-month return"),
    "carry_3m1m":  ("lambda s: (s.shift(21)/s.shift(63)-1.0) - (s/s.shift(21)-1.0)",
                    "3m-1m carry proxy: 3m return (ex last month) minus last-month return"),
    "carry_12m3m": ("lambda s: (s.shift(63)/s.shift(252)-1.0) - (s/s.shift(63)-1.0)",
                    "12m-3m carry proxy: 12m return (ex last quarter) minus last-quarter return"),
}

EXPR = {
    "carry_12m1m": "close.shift(21)/close.shift(252)-1.0 - (close/close.shift(21)-1.0)",
    "carry_6m1m":  "close.shift(21)/close.shift(126)-1.0 - (close/close.shift(21)-1.0)",
    "carry_3m1m":  "close.shift(21)/close.shift(63)-1.0 - (close/close.shift(21)-1.0)",
    "carry_12m3m": "close.shift(63)/close.shift(252)-1.0 - (close/close.shift(63)-1.0)",
}

signals = {}
print("\n=== per-asset own-calendar construction ===")
for fid, (lam, desc) in CAND.items():
    sig = per_asset(panel, eval(lam)).reindex(index=panel.index, columns=panel.columns)
    signals[fid] = sig
    print(f"  {fid:14s} shape={sig.shape} nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# sanity: what does the factor look like for a couple of assets (XAU backwardation episodes)
print("\n=== sample levels (XAU carry_12m1m, tail 5) ===")
print(signals["carry_12m1m"]["XAU"].dropna().tail(5).round(4).to_string())

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n=== validation (admission h=10) ===")
library = load_library_signals(panel)
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5]
print(f"\nPASSERS (gate + robustness): {passers}")

# regime breakdown for passers
print("\n=== regime breakdown for passers (IC10 / ICIR10 / n) ===")
regime_out = {}
for fid in passers:
    sig = signals[fid]
    rd = {}
    parts = [fid]
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            parts.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n_dates": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(parts))

json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                       for k, v in results.items()},
           "passers": passers, "regime": regime_out,
           "signals_expr": EXPR, "signals_desc": {k: v[1] for k, v in CAND.items()}},
          open("scripts/_miner1_cycle23_carry_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle23_carry_results.json")
print("DONE")
