"""miner_3 novel factor exploration at 2032-05-03 (visible through 2032-04-30).
Monitors only -- does NOT touch the live account. No lookahead.
Tests novel cross-asset / risk / breadth / momentum variants, reports
full + recent-window IC/ICIR and per-year breakdown.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE,
                          ICIR_GATE, max_lib_corr)

END = "2032-04-30"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
fwd = forward_ret(close, 10)
lib_panels = library_panel(close, macro)
c = close
vix = macro["VIX"]; dxy = macro["DXY"]

cands = {}

# 1. USDJPY-conditional beta (risk-on JPY weakening)
usdjpy = macro["USDJPY"]
r_usdjpy = usdjpy.pct_change(); usdjpy_20 = usdjpy / usdjpy.shift(20) - 1.0
jp_beta = ret.rolling(60, min_periods=30).cov(r_usdjpy) / r_usdjpy.rolling(60, min_periods=30).var()
cands["jpy_beta_cond_60x20"] = (-jp_beta.multiply(usdjpy_20, axis=0))

# 2. VIX-change-conditional beta (only meaningful in rising-vol regimes)
vixr = vix.pct_change(); vix_up = (vix / vix.shift(20) - 1.0).where(vix / vix.shift(20) - 1.0 > 0, 0.0)
vix_beta = ret.rolling(60, min_periods=30).cov(vixr) / vixr.rolling(60, min_periods=30).var()
cands["vix_beta_cond_up_60x20"] = (-vix_beta.multiply(vix_up, axis=0))

# 3. US10Y relative momentum (demeaned bond trend)
mom10y = c["US10Y"] / c["US10Y"].shift(20) - 1.0
cands["us10y_mom_20"] = (c / c.shift(20) - 1.0).sub(mom10y, axis=0)

# 4. Downside-vol ratio 40d
neg = ret.where(ret < 0, 0.0)
ds40 = (neg ** 2).rolling(40).mean().apply(np.sqrt)
cands["downside_vol_ratio_40"] = -(ds40 / ret.rolling(40).std())

# 5. Distance from 60d high (mean-reversion)
cands["dd_60d"] = (c / c.rolling(60).max() - 1.0)

# 6. Risk-adjusted momentum 60d
mom60 = c / c.shift(60) - 1.0
cands["risk_adj_mom_60"] = mom60 / ret.rolling(60).std()

# 7. Gross momentum 60d skip5
cands["mom_60d_skip5"] = (c.shift(5) / c.shift(5 + 60) - 1.0)

# 8. Cross-sectional breadth above MA20
above = (c > c.rolling(20).mean()).astype(float)
cands["above_ma20_breadth"] = above - above.mean(axis=1)

# 9. 90d down-vol ratio (longer-tail defensive)
ds90 = (neg ** 2).rolling(90).mean().apply(np.sqrt)
cands["downside_vol_ratio_90"] = -(ds90 / ret.rolling(90).std())

# 10. skewness 30d skip5
cands["skew_30d_skip5"] = ret.shift(5).rolling(30).skew()

# 11. realized-vol z-score (vol of vol, 20x60)
vol20 = ret.rolling(20).std()
cands["vol_z_20x60"] = (vol20 - vol20.rolling(60).mean()) / vol20.rolling(60).std()

# 12. eurusd conditional beta
eurusd = macro["EURUSD"]; r_eur = eurusd.pct_change(); eu20 = eurusd / eurusd.shift(20) - 1.0
eu_beta = ret.rolling(60, min_periods=30).cov(r_eur) / r_eur.rolling(60, min_periods=30).var()
cands["eurusd_beta_cond_60x20"] = eu_beta.multiply(eu20, axis=0)

print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}\n")
print(f"{'factor':24s}{'IC10':>8}{'ICIR':>8}{'hit':>6}{'n':>6} | {'IC1y':>7}{'ICIR1y':>7}{'IC2y':>7} | {'turn':>6}{'covD8':>6} | {'maxrho':>7}")
results = {}
for name in cands:
    f = cands[name]
    ic_s = daily_ic(f, fwd)
    st = ic_stats(ic_s, 10)
    ser = ic_s.dropna()
    r1 = ser[ser.index >= (ser.index.max() - np.timedelta64(365, 'D'))]
    r2 = ser[ser.index >= (ser.index.max() - np.timedelta64(730, 'D'))]
    ic1 = r1.mean() if len(r1) else np.nan
    icir1 = (r1.mean() / r1.std(ddof=1)) if len(r1) > 2 else np.nan
    ic2 = r2.mean() if len(r2) else np.nan
    turn = rank_turnover(f, 10)
    cov = coverage_stats(f, fwd)
    rho, _ = max_lib_corr(f, lib_panels)
    gate = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
    results[name] = st
    flag = "PASS" if gate else ""
    print(f"{name:24s}{st['ic']:+.4f} {st['icir']:+.3f} {st['hit']:5.2f} {st['n']:5d} | "
          f"{ic1:+.4f} {icir1:+.3f} {ic2:+.4f} | {turn:6.2f} {cov['coverage_dates_ge8']:5.2f} | {rho:.3f} {flag}")

print("\nPer-year h10 IC (direction-adjusted):")
for name in cands:
    ic = daily_ic(cands[name], fwd).dropna()
    parts = []
    for yr in range(2025, 2033):
        sub = ic[ic.index.year == yr]
        m = sub.mean() if len(sub) else np.nan
        parts.append(f"{yr}:{m:+.3f}(n={len(sub)})")
    print(f"{name:24s} " + "  ".join(parts))