"""miner_3 cycle: screen round 6 - novel factor families.
Families: variance ratio (trend vs reversion), rolling Sharpe (quality),
trend R2 (trend consistency), gap-risk share (overnight vol), leverage effect,
conditional betas on crypto/copper/USDJPY, volume-price correlation.
Validation window: 2020-01-01 .. 2026-07-15 (warm-up). Horizon-10 admission gate:
|IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
from miner3_lib import (WATCH, MACRO, MIN_ASSETS_PER_DATE, load_panel, load_macro,
                        fwd_returns, rank_ic_series, turnover_10d_rank,
                        _load_library_factors, validate_factor)
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

# ---- OHLCV panel loader ----
def load_ohlcv(days: int = 4000):
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is None or not len(df):
            continue
        idx = pd.to_datetime(df["date"])
        closes[s] = df.set_index(idx)["close"].astype(float)
        opens[s] = df.set_index(idx)["open"].astype(float)
        highs[s] = df.set_index(idx)["high"].astype(float)
        lows[s] = df.set_index(idx)["low"].astype(float)
        vols[s] = df.set_index(idx)["volume"].astype(float)
    def tidy(d):
        p = pd.concat(d, axis=1, sort=True)
        return p[~p.index.duplicated(keep="last")].sort_index()
    return tidy(closes), tidy(opens), tidy(highs), tidy(lows), tidy(vols)

close, open_, high, low, volume = load_ohlcv()
print(f"close panel: {close.shape}, {close.index.min().date()} .. {close.index.max().date()}")

def per_asset_ohlc(fn):
    def wrapper(panel_close, macro):
        cols = {}
        for a in panel_close.columns:
            s = panel_close[a].dropna()
            o = open_[a].reindex(s.index)
            h = high[a].reindex(s.index)
            l = low[a].reindex(s.index)
            v = volume[a].reindex(s.index)
            cols[a] = fn(s, o, h, l, v)
        return pd.DataFrame(cols, index=panel_close.index)
    return wrapper

# ---- candidate 1: variance ratio VR(5) using 20d estimation window ----
def cand_variance_ratio_20x5():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        r = s.pct_change()
        var1 = r.rolling(20).var()
        var5 = s.pct_change(5).rolling(20).var()
        return var5 / (5.0 * var1)
    return f(close, load_macro())

# ---- candidate 2: rolling Sharpe ratio 60d ----
def cand_roll_sharpe_60():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        r = s.pct_change()
        return r.rolling(60).mean() / r.rolling(60).std()
    return f(close, load_macro())

# ---- candidate 3: trend R2 over 60d (OLS of log price on time) ----
def cand_trend_r2_60():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        lp = np.log(s)
        x = np.arange(len(s))
        r2 = lp.rolling(60).apply(
            lambda y: np.corrcoef(x[:len(y)], y)[0, 1] ** 2 if len(y) == 60 and np.isfinite(y).all() else np.nan,
            raw=True)
        return r2
    return f(close, load_macro())

# ---- candidate 4: gap-risk share (std of overnight ret / std of total ret) 20d ----
def cand_gap_vol_share_20():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        prev_close = s.shift(1)
        overnight = o / prev_close - 1.0
        total = s.pct_change()
        return overnight.rolling(20).std() / total.rolling(20).std()
    return f(close, load_macro())

# ---- candidate 5: leverage effect (corr of return with lagged 20d vol) ----
def cand_lever_effect_60():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        r = s.pct_change()
        vol = r.rolling(20).std().shift(1)
        return r.rolling(60).corr(vol)
    return f(close, load_macro())

# ---- candidate 6: conditional beta on crypto basket (BTC+ETH) ----
def cand_crypto_beta_cond_60x20():
    m = load_macro()
    panel = load_panel()
    crypto = panel[["BTC", "ETH"]].dropna(how="all")
    basket = crypto.mean(axis=1)
    br = basket.pct_change()
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), br.reindex(s.index).rename("b")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["b"]) / z["b"].rolling(60).var()
        bret = (basket / basket.shift(20) - 1.0).reindex(s.index)
        cols[a] = beta * bret
    return pd.DataFrame(cols, index=panel.index)

# ---- candidate 7: conditional beta on COPPER ----
def cand_copper_beta_cond_60x20():
    panel = load_panel()
    cu = panel["COPPER"].dropna()
    cr = cu.pct_change()
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), cr.reindex(s.index).rename("c")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["c"]) / z["c"].rolling(60).var()
        cret = (cu / cu.shift(20) - 1.0).reindex(s.index)
        cols[a] = beta * cret
    return pd.DataFrame(cols, index=panel.index)

# ---- candidate 8: conditional beta on USDJPY ----
def cand_usdjpy_beta_cond_60x20():
    m = load_macro()
    jpy = m["USDJPY"].dropna()
    jr = jpy.pct_change()
    panel = load_panel()
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), jr.reindex(s.index).rename("j")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["j"]) / z["j"].rolling(60).var()
        jret = (jpy / jpy.shift(20) - 1.0).reindex(s.index)
        cols[a] = beta * jret
    return pd.DataFrame(cols, index=panel.index)

# ---- candidate 9: volume-price correlation 20d ----
def cand_vol_price_corr_20():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        r = s.pct_change()
        dv = v.pct_change()
        return r.rolling(20).corr(dv)
    return f(close, load_macro())

# ---- candidate 10: drawdown depth (distance from running max) 60d ----
def cand_dd_depth_60():
    @per_asset_ohlc
    def f(s, o, h, l, v):
        return s / s.rolling(60).max() - 1.0
    return f(close, load_macro())

CANDIDATES = {
    "variance_ratio_20x5": lambda p, m: cand_variance_ratio_20x5(),
    "roll_sharpe_60": lambda p, m: cand_roll_sharpe_60(),
    "trend_r2_60": lambda p, m: cand_trend_r2_60(),
    "gap_vol_share_20": lambda p, m: cand_gap_vol_share_20(),
    "lever_effect_60": lambda p, m: cand_lever_effect_60(),
    "crypto_beta_cond_60x20": lambda p, m: cand_crypto_beta_cond_60x20(),
    "copper_beta_cond_60x20": lambda p, m: cand_copper_beta_cond_60x20(),
    "usdjpy_beta_cond_60x20": lambda p, m: cand_usdjpy_beta_cond_60x20(),
    "vol_price_corr_20": lambda p, m: cand_vol_price_corr_20(),
    "dd_depth_60": lambda p, m: cand_dd_depth_60(),
}

RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        RESULTS[name] = validate_factor(name, fn)
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")

print("\n===== SUMMARY (h10 gate |IC|>=0.007, |ICIR|>=0.084, corr<0.5) =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    passed = abs(r["ic_h10"]) >= 0.007 and abs(r["icir_h10"]) >= 0.084
    lowcorr = r.get("max_abs_library_correlation", 1.0) < 0.5
    print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r.get('max_abs_library_correlation', float('nan')):.3f} "
          f"-> {'PASS' if passed else 'FAIL'} {'corr-ok' if lowcorr else 'CORR-HI'}")
