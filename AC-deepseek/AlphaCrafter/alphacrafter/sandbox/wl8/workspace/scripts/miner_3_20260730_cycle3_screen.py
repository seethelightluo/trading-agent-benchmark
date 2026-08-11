"""miner_3 2026-07-30 cycle3: batch screen of orthogonal factor families.
Families: calendar seasonality, volume dynamics, price-location, efficiency, trend-quality.
Uses the 15-asset tradable universe; IC at 10d primary horizon; checks Spearman rho
against the ACTIVE library (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys, importlib.util

spec = importlib.util.spec_from_file_location("common", "scripts/miner_3_20260730_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
WATCH = common.WATCH
load_data = common.load_data
factor_ic_table = common.factor_ic_table
rank_turnover = common.rank_turnover

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
rets = {a: c.pct_change() for a, c in closes.items()}
vols = {a: d["volume"].astype(float) for a, d in data.items()}

# ---------- active library panels (with yield-beta gatekeeper) ----------
vix = None
try:
    vdf = pd.read_csv("../persistent/index_data/VIX.csv")
    vdf["date"] = pd.to_datetime(vdf["date"])
    vix = vdf.set_index("date").sort_index()["close"].astype(float)
except Exception as e:
    print("vix load fail", e)

lib = {}
for a, c in closes.items():
    lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
    if vix is not None:
        r = c.pct_change()
        beta = r.rolling(60).cov(vix.pct_change()) / vix.pct_change().rolling(60).var()
        lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta * (vix / vix.shift(20) - 1.0)
    r = c.pct_change()
    # yield beta: beta of asset ret to US10Y ret over 60d x 20d yield move
    y = closes.get("US10Y")
    if y is not None:
        yr = y.pct_change()
        beta_y = r.rolling(60).cov(yr) / yr.rolling(60).var()
        ymove = y / y.shift(20) - 1.0
        lib.setdefault("yield_beta_cond_60x20", {})[a] = beta_y * ymove
lib_panels = {k: pd.DataFrame(v) for k, v in lib.items()}
print("[lib] panels:", {k: int(v.notna().sum().sum()) for k, v in lib_panels.items()})

# ---------- candidate factors ----------
cands = {}

def add(fid, fdict):
    cands[fid] = fdict

# Calendar seasonality
# 1) same-weekday average return over last 10 occurrences (weekday seasonality)
for a, c in closes.items():
    s = c.copy()
    dow = s.index.dayofweek
    r = c.pct_change()
    f = pd.Series(np.nan, index=s.index)
    for wd in range(5):
        mask = dow == wd
        idx = s.index[mask]
        vals = r[mask]
        f.loc[idx] = vals.rolling(10, min_periods=5).mean()
    add("wday_ret_10", {a: f})

# 2) turn-of-month effect: return over last 5d leading into month end
for a, c in closes.items():
    r = c.pct_change()
    tom = pd.Series(np.nan, index=c.index)
    # days within 5 of month end
    nxt_month_start = c.index.to_series().shift(-1).dt.month
    is_month_end = nxt_month_start.ne(c.index.to_series().dt.month).astype(int)
    tom = r.rolling(5).sum() * is_month_end.replace(0, np.nan)
    add("turn_of_month_5", {a: tom})

# Volume dynamics
# 3) volume trend: 20d mean volume / 60d mean volume
add("vol_trend_20x60", {a: vols[a].rolling(20).mean() / vols[a].rolling(60).mean() for a in closes})
# 4) volume z-score 20d
add("vol_z_20", {a: (vols[a] - vols[a].rolling(20).mean()) / vols[a].rolling(20).std() for a in closes})
# 5) price-volume divergence: 20d return signed by volume trend (confirmation)
add("pv_conf_20", {a: (closes[a].pct_change(20)) * np.sign(vols[a].rolling(20).mean() / vols[a].rolling(60).mean() - 1.0) for a in closes})

# Price-location
# 6) position in 20d high-low range
def hl_pos(win):
    out = {}
    for a, c in closes.items():
        hi = c.rolling(win).max(); lo = c.rolling(win).min()
        out[a] = (c - lo) / (hi - lo).replace(0, np.nan)
    return out
add("hl_pos_20", hl_pos(20))
add("hl_pos_60", hl_pos(60))
# 7) distance from 60d high (negative = below high)
add("dist_high_60", {a: closes[a] / closes[a].rolling(60).max() - 1.0 for a in closes})
# 8) short-term range normalized by long-term range (range squeeze)
add("range_squeeze_10x60", {a: ((closes[a].rolling(10).max() - closes[a].rolling(10).min()) / closes[a]) /
                            ((closes[a].rolling(60).max() - closes[a].rolling(60).min()) / closes[a]) for a in closes})

# Efficiency / trend quality
# 9) Kaufman efficiency ratio 20d
add("eff_ratio_20", {a: (closes[a] - closes[a].shift(20)).abs() / (rets[a].abs().rolling(20).sum()).replace(0, np.nan) for a in closes})
# 10) Kaufman efficiency ratio 40d
add("eff_ratio_40", {a: (closes[a] - closes[a].shift(40)).abs() / (rets[a].abs().rolling(40).sum()).replace(0, np.nan) for a in closes})
# 11) linear-trend R2 over 60d (trend straightness)
def trend_r2(win):
    out = {}
    for a, c in closes.items():
        lc = np.log(c)
        r2 = lc.rolling(win).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] ** 2 * (len(x) * x.var()) / ((x - x.mean()) ** 2).sum() if len(x) == win else np.nan, raw=True)
        out[a] = r2
    return out
# simpler robust version
def trend_r2b(win):
    out = {}
    for a, c in closes.items():
        lc = np.log(c)
        def _r2(x):
            if len(x) != win or np.std(x) == 0:
                return np.nan
            t = np.arange(win)
            b = np.polyfit(t, x, 1)
            pred = b[0] * t + b[1]
            ss_res = np.sum((x - pred) ** 2)
            ss_tot = np.sum((x - x.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out[a] = lc.rolling(win).apply(_r2, raw=True)
    return out
add("trend_r2_60", trend_r2b(60))

# Cross-asset: rolling 60d correlation of asset to XAU (safe-haven/commodity linkage)
# 12) gold correlation 60d
add("gold_corr_60", {a: rets[a].rolling(60).corr(rets["XAU"]) for a in closes})
# 13) equity-breadth linkage: correlation of asset to SPX over 60d
add("spx_corr_60", {a: rets[a].rolling(60).corr(rets["SPX"]) for a in closes})

# ---------- screen ----------
print("\n" + "=" * 100)
print(f"{'factor':<20}{'IC10':>8}{'ICIR10':>8}{'hit':>7}{'n':>6}{'cov':>7}{'to':>7}{'rho_mom':>8}{'rho_vix':>8}{'rho_yld':>8}")
rows = []
for fid, fdict in cands.items():
    fdf = pd.DataFrame(fdict)
    t = factor_ic_table(fdict, data, horizons=(10,), min_assets=8, primary_h=10)[10]
    if t is None:
        print(f"{fid:<20} degenerate")
        continue
    to = rank_turnover(fdict)
    # spearman vs library panels (pooled)
    fstack = fdf.stack()
    fstack = fstack[fstack.notna()]
    rhos = {}
    for lid, lp in lib_panels.items():
        lstack = lp.stack()
        lstack = lstack[lstack.notna()]
        both = fstack.index.intersection(lstack.index)
        if len(both) < 100:
            rhos[lid] = np.nan
        else:
            rhos[lid], _ = spearmanr(fstack.loc[both].values, lstack.loc[both].values)
    rows.append((fid, t, to, rhos))
    print(f"{fid:<20}{t['ic']:>8.4f}{t['icir']:>8.4f}{t['ic_hit']:>7.3f}{t['n_dates']:>6}"
          f"{t['dates_ge8']:>7.2f}{to:>7.2f}"
          f"{rhos.get('mom_10d_skip5', np.nan):>8.3f}{rhos.get('vix_beta_cond_60x20', np.nan):>8.3f}"
          f"{rhos.get('yield_beta_cond_60x20', np.nan):>8.3f}")

print("\n[gate check] PASS_IC = |IC|>=0.0070 and |ICIR|>=0.0840; rho_yld < 0.5 required")
for fid, t, to, rhos in rows:
    gate_ic = abs(t["ic"]) >= 0.0070 and abs(t["icir"]) >= 0.0840
    rho_yld = abs(rhos.get("yield_beta_cond_60x20", 0) or 0)
    rho_vix = abs(rhos.get("vix_beta_cond_60x20", 0) or 0)
    rho_mom = abs(rhos.get("mom_10d_skip5", 0) or 0)
    ok = gate_ic and rho_yld < 0.5 and rho_vix < 0.5 and rho_mom < 0.5
    print(f"  {fid:<20} gate={'PASS' if ok else 'FAIL':<4} IC={t['ic']:.4f} ICIR={t['icir']:.4f} "
          f"rho_yld={rho_yld:.3f} rho_vix={rho_vix:.3f} rho_mom={rho_mom:.3f}")
