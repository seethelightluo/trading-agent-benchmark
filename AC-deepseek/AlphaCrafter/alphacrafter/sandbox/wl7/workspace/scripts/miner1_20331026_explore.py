"""miner_1 candidate factor exploration, visible end 2033-10-26.

Tests candidate ideas vs h=10 forward returns on the 15-asset cross-asset
universe. Reports IC/ICIR/hit, coverage, turnover, and max abs library
correlation vs the active library. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
No lookahead. Macro (DXY/EURUSD/VIX) are observation-only signals.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2033-10-26"
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)

cands = {}

# C1: realized vol 60d (low-vol tilt); sign - so lower vol = higher score? test raw first
cands["vol60_neg"] = -(ret.rolling(60).std())

# C2: volatility-of-volatility 20x60 (2nd order vol)
vol20 = ret.rolling(20).std()
cands["volofvol_20x60"] = vol20.rolling(60).std()

# C3: trend persistence / R2 of 60d linear trend (efficiency, acceleration)
def r2_60(x):
    n = len(x)
    if n < 40:
        return np.nan
    t = np.arange(n)
    m, b = np.polyfit(t, x, 1)
    pred = m * t + b
    ss_res = ((x - pred) ** 2).sum()
    ss_tot = ((x - x.mean()) ** 2).sum() + 1e-12
    return ss_res / ss_tot if ss_tot > 0 else np.nan
# r2 (fit quality); higher = smoother trend
cands["r2_fit_60"] = close.rolling(60).apply(lambda x: r2_60(x.values), raw=True)

# C4: 1d autocorrelation / efficiency ratio 10d / 60d (trend vs noise)
abs_sum10 = ret.abs().rolling(10).sum()
abs_ret10 = (close / close.shift(10) - 1).abs()
cands["eff_ratio_10x60"] = (close / close.shift(60) - 1).abs() / (ret.abs().rolling(60).sum() + 1e-12)

# C5: 5d momentum vs 60d momentum relative (short-term acceleration) demeant cross-sectionally
mom5 = close / close.shift(10) - 1.0
mom60 = close / close.shift(60) - 1.0
cands["accel_5_60_rel"] = (mom5 - mom60).subtract((mom5 - mom60).median(axis=1), axis=0)

# C6: dispersion contribution / cross-sectional beta to 20d equal-weight, direction vs 60d beta change
ret20 = ret.rolling(20).mean()
mkt = ret.mean(axis=1)
beta20 = ret.rolling(20).cov(mkt) / mkt.rolling(20).var()
beta120 = ret.rolling(120).cov(mkt) / mkt.rolling(120).var()
cands["beta_change_20_120"] = beta20 - beta120

# C7: relative 60d vol (asset vol / cross-sectional median vol) - low relative vol
rel_vol = ret.rolling(60).std()
cands["rel_vol_60"] = -(rel_vol.subtract(rel_vol.median(axis=1), axis=0))

# C8: 20d kurtosis (tail asymmetry) - may already exist as kurt_20d_skip5, try raw 60d kurtosis
cands["kurt_60"] = ret.rolling(60).kurt()

# C9: drawdown depth 60d (positive when near high)
cands["dd_dist_60"] = close / close.rolling(60).max() - 1.0

# C10: XAU/COPPER risk-off-offence rotation relative factor using macro? skip (uses observation assets directly gives 0 variance across them) - test asset mom vs DXY beta
# Instead: composite commodity momentum (WTI) leading energy cross-section is meaningless (1 asset). skip.

def max_lib_corr(cand, lib_panels):
    flat = cand.stack()
    best = 0.0; pairs = {}
    for name, p in lib_panels.items():
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"]))
        pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs

print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}  horizon=10")
print(f"{'candidate':18s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covD8':>6s} {'turn':>6s} {'maxrho':>7s}  GATE")
res = {}
for name, panel in cands.items():
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:18s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} {cov['coverage_dates_ge8']:6.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    s = ic.dropna()
    for lab, w in (("1y",365), ("2y",730)):
        rs = s[s.index >= s.index.max() - np.timedelta64(w, "D")]
        if len(rs):
            m = rs.mean(); sd = rs.std(ddof=1)
            print(f"    {lab}: IC {m:+.4f} ICIR {m/sd if sd>0 else float('nan'):.3f} hit {(rs>0).mean():.2f} n={len(rs)}")
    res[name] = dict(ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     max_abs_library_correlation=mrho, pairs=pairs, gate=gate)
json.dump(res, open("scripts/miner1_20331026_explore.json", "w"), indent=1, default=str)