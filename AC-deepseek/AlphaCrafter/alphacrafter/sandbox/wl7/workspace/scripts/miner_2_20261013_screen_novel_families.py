"""miner_2 screening: novel factor families (fast vectorized version).
Universe: 15 tradable cross-asset instruments. Warm-up factor window
2020-01-01..2026-07-15 (data visible through 2026-07-29).
Admission gates: |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_2_lib import (load_panel, load_macro, fwd_returns, rank_ic_series,
                         turnover_10d_rank, library_signals, library_corr,
                         FACTOR_LAST, MIN_ASSETS, ADMISSION, per_asset)

t0 = time.time()
HORIZONS = (1, 2, 3, 5, 10, 20)
panel = load_panel()
macro = load_macro()
rets = panel.pct_change()

# ---------- candidate factor definitions (all novel vs library) ----------
cands = {}

# 1) Rolling skewness of 20d returns
@per_asset
def skew_20(s):
    r = s.pct_change()
    return r.rolling(20).skew()
cands["skew_20d"] = skew_20(panel, macro)

# 2) Return autocorrelation at lag 5 over 60d window (vectorized)
@per_asset
def autocorr_5_60(s):
    r = s.pct_change()
    r5 = r.shift(5)
    mu = r.rolling(60).mean()
    mu5 = r5.rolling(60).mean()
    num = ((r - mu) * (r5 - mu5)).rolling(60).mean()
    den = r.rolling(60).std() * r5.rolling(60).std()
    return num / den
cands["autocorr_5_60"] = autocorr_5_60(panel, macro)

# 3) Intraday close location: mean of (close-low)/(high-low) over 20d
def intraday_pos_20(panel, macro):
    cols = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= "2026-07-29"].set_index("date").sort_index()
        hl = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        cols[a] = hl.rolling(20).mean()
    return pd.DataFrame(cols, index=panel.index)
cands["intraday_pos_20"] = intraday_pos_20(panel, macro)

# 4) Intraday range ratio: mean of (high-low)/close over 20d
def range_ratio_20(panel, macro):
    cols = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= "2026-07-29"].set_index("date").sort_index()
        rr = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        cols[a] = rr.rolling(20).mean()
    return pd.DataFrame(cols, index=panel.index)
cands["range_ratio_20"] = range_ratio_20(panel, macro)

# 5) Return/vol ratio (Sharpe-like) 20d
@per_asset
def sharpe_20(s):
    r = s.pct_change()
    return r.rolling(20).mean() / r.rolling(20).std()
cands["sharpe_20"] = sharpe_20(panel, macro)

# 6) Distance from 120d high (drawdown distance)
cands["dist_high_120"] = panel / panel.rolling(120).max() - 1.0

# 7) RSI(14)
@per_asset
def rsi_14(s):
    r = s.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
cands["rsi_14"] = rsi_14(panel, macro)

# 8) Momentum acceleration: mom20 - mom60 (both skip 5)
mom20 = panel.shift(5) / panel.shift(25) - 1.0
mom60 = panel.shift(5) / panel.shift(65) - 1.0
cands["mom_accel_20x60"] = mom20 - mom60

# 9) Up/down day return asymmetry over 20d
@per_asset
def updown_ratio_20(s):
    r = s.pct_change()
    return r.clip(lower=0).rolling(20).mean() / (-r.clip(upper=0)).rolling(20).mean().replace(0, np.nan)
cands["updown_ratio_20"] = updown_ratio_20(panel, macro)

# 10) Z-score of close vs 60d SMA
@per_asset
def zscore_60(s):
    return (s - s.rolling(60).mean()) / s.rolling(60).std()
cands["zscore_60"] = zscore_60(panel, macro)

# 11) Trend strength: SMA20/SMA60 - 1
@per_asset
def trend_ma_20x60(s):
    return s.rolling(20).mean() / s.rolling(60).mean() - 1.0
cands["trend_ma_20x60"] = trend_ma_20x60(panel, macro)

# 12) Conditional beta on XAU (gold) - flight-to-safety regime
def gold_beta_cond_60x20(panel, macro):
    xaur = panel["XAU"].pct_change()
    out = {}
    for a in panel.columns:
        r = panel[a].pct_change()
        beta = r.rolling(60).cov(xaur) / xaur.rolling(60).var()
        out[a] = beta * (panel["XAU"] / panel["XAU"].shift(20) - 1.0)
    return pd.DataFrame(out, index=panel.index)
cands["gold_beta_cond_60x20"] = gold_beta_cond_60x20(panel, macro)

# 13) Conditional beta on WTI (oil) - commodity cycle regime
def oil_beta_cond_60x20(panel, macro):
    wtir = panel["WTI"].pct_change()
    out = {}
    for a in panel.columns:
        r = panel[a].pct_change()
        beta = r.rolling(60).cov(wtir) / wtir.rolling(60).var()
        out[a] = beta * (panel["WTI"] / panel["WTI"].shift(20) - 1.0)
    return pd.DataFrame(out, index=panel.index)
cands["oil_beta_cond_60x20"] = oil_beta_cond_60x20(panel, macro)

# 14) Vol regime: 20d vol vs its 60d median
@per_asset
def vol_regime_20x60(s):
    v = s.pct_change().rolling(20).std()
    return v / v.rolling(60).median() - 1.0
cands["vol_regime_20x60"] = vol_regime_20x60(panel, macro)

print(f"factors built in {time.time()-t0:.1f}s", flush=True)

fwd = {h: fwd_returns(panel, h) for h in HORIZONS}
libs = library_signals(panel)

rows = []
for name, f in cands.items():
    fw = f.loc[:FACTOR_LAST]
    ic10 = rank_ic_series(fw, fwd[10])
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    ic10a = ic10 * direction
    icir10 = float(ic10a.mean() / ic10a.std()) if len(ic10a) > 2 and ic10a.std() > 0 else np.nan
    ic5 = rank_ic_series(fw, fwd[5]) * direction
    icir5 = float(ic5.mean() / ic5.std()) if len(ic5) > 2 and ic5.std() > 0 else np.nan
    ic20 = rank_ic_series(fw, fwd[20]) * direction
    icir20 = float(ic20.mean() / ic20.std()) if len(ic20) > 2 and ic20.std() > 0 else np.nan
    valid = fw.notna()
    cov_ad = float(valid.mean().mean())
    cov_d8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    to = turnover_10d_rank(fw)
    max_corr, per = library_corr(fw, panel, libs)
    gate = abs(ic10a.mean()) >= ADMISSION["ic"] and abs(icir10) >= ADMISSION["icir"]
    rows.append({
        "name": name, "dir": round(direction, 2),
        "ic10": round(float(ic10a.mean()), 4), "icir10": round(icir10, 4),
        "ic5": round(float(ic5.mean()), 4), "icir5": round(icir5, 4),
        "ic20": round(float(ic20.mean()), 4), "icir20": round(icir20, 4),
        "hit10": round(float((ic10a > 0).mean()), 3), "n10": len(ic10a),
        "cov_ad": round(cov_ad, 3), "cov_d8": round(cov_d8, 3),
        "turn": round(to, 3), "maxlib": round(max_corr, 3),
        "verdict": "PASS" if gate else "fail"
    })
    print(f"done {name} ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame(rows).sort_values("ic10", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
print("\nGATE: |IC10|>=0.007 and |ICIR10|>=0.084")
