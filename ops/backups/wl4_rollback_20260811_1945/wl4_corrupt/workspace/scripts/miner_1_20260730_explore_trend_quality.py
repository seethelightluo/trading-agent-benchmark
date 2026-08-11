"""miner_1 trend-quality family exploration (2026-07-30, visible through 2026-07-29).

Idea: trend QUALITY (strength/smoothness of the trend, not its direction) should
predict continuation across cross-asset instruments. Momentum direction factors
are already in the library; magnitude/smoothness measures are decorrelated.
Candidates: ADX(14), Kaufman efficiency ratio, risk-adjusted momentum,
recovery speed (5d bounce vs 60d drawdown), trend consistency.
Gates: |IC|>=0.007, |ICIR|>=0.084 at h=10; library corr <= 0.5.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr, full_eval, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
H = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE}, axis=1).sort_index()
L = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE}, axis=1).sort_index()

cands = {}

# --- ADX(14): Wilder directional movement index, trend strength magnitude ---
def wilder_adx(H, L, C, n=14):
    up = H.diff()
    dn = -L.diff()
    plus_dm = pd.DataFrame(np.where((up > dn) & (up > 0), up, 0.0), index=H.index, columns=H.columns)
    minus_dm = pd.DataFrame(np.where((dn > up) & (dn > 0), dn, 0.0), index=H.index, columns=H.columns)
    tr = pd.concat([(H - L).abs(), (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1)
    tr = tr.groupby(tr.columns, axis=1).max()
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx

cands["adx_14"] = wilder_adx(H, L, closes, 14)

# --- Kaufman efficiency ratio: |close-close.shift(n)| / sum(|rets|, n) ---
for n in (20, 60):
    num = (closes - closes.shift(n)).abs()
    den = rets.abs().rolling(n).sum()
    cands[f"eff_ratio_{n}"] = (num / den).where(den > 1e-12)

# --- risk-adjusted momentum: mom_60d_skip5 / vol_20 (vol-scaled trend) ---
cands["mom_vol_60x20"] = (closes.shift(5) / closes.shift(65) - 1.0) / rets.rolling(20).std()

# --- recovery speed: 5d bounce / 60d drawdown from peak ---
dd60 = closes / closes.rolling(60).max() - 1.0
cands["dd_speed_5x60"] = (closes / closes.shift(5) - 1.0) / dd60.abs().clip(lower=1e-6)

# --- trend consistency: mean(sign of 20d returns) over 60d window ---
sign20 = np.sign(closes.pct_change(20))
cands["trend_consist_20x60"] = sign20.rolling(3).mean()

# --- distance from 20d high (pullback depth) ---
cands["pullback_20"] = closes / closes.rolling(20).max() - 1.0

# --- evaluate ---
lib = library_signals(panels, closes, rets)
rows = []
for name, panel in cands.items():
    ics = rank_ic_series(panel, forward_returns(closes, 10), 8)
    if len(ics) < 200:
        print(f"{name:24s} NOTE insufficient n_ic_dates={len(ics)}")
        continue
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    corr, key = max_library_corr(panel, lib)
    m["max_lib_corr"] = corr
    m["max_corr_key"] = key
    m["name"] = name
    m["decay"] = decay_profile(panel, closes, (1, 3, 5, 10, 20), 8, 1)
    rows.append(m)
    flag = "PASS" if (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and corr <= 0.5) else ""
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) decay={m['decay']} {flag}")

df = pd.DataFrame(rows).set_index("name")
df.to_csv("scripts/_tq_results.csv")
print("\n=== PASS (|ic|>=0.007 & |icir|>=0.084 & rho<=0.5) ===")
passing = df[(df.ic.abs() >= 0.007) & (df.icir.abs() >= 0.084) & (df.max_lib_corr <= 0.5)]
print(passing[["ic", "icir", "n_ic_dates", "coverage_asset_days", "max_lib_corr", "max_corr_key"]].to_string()
      if len(passing) else "none")
print(f"assets used: {len(closes.columns)}, dates: {len(closes)}")
