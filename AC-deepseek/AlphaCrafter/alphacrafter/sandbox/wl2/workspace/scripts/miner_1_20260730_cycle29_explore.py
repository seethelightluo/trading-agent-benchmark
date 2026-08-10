"""Cycle 29 exploration: orthogonal-to-momentum factor families.

Active library (2 factors): mom20_volproxy60, dxy_beta_cond_60x20.
Goal: find candidates from risk / microstructure / statistical families that
pass IC>=0.007, |ICIR|>=0.084 at 10d horizon and stay orthogonal (|rho|<0.5)
to BOTH active factors.
"""
import sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, panel_rank_corr,
                         turnover_rank, coverage_stats, validate_factor,
                         load_library_signals)

panel = load_panel()
close = panel

# ---- Rebuild the 2 ACTIVE library signals exactly ----
# mom20_volproxy60: (close.shift(5)/close.shift(25)-1) * 1/(1+|close.shift(5)/close.shift(65)-1|)
mom60_proxy = per_asset(close, lambda s: s.shift(5) / s.shift(65) - 1.0)
damp = 1.0 / (1.0 + mom60_proxy.abs())
mom20_raw = per_asset(close, lambda s: s.shift(5) / s.shift(25) - 1.0)
sig_mom = mom20_raw * damp

# dxy_beta_cond_60x20: 60d beta of asset ret on DXY ret * 20d DXY momentum
dxy = macro_series("DXY")
dxy_ret = dxy.pct_change()
dxy_20 = dxy / dxy.shift(20) - 1.0
beta_parts = {}
for a in close.columns:
    s = close[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), dxy_ret.reindex(ar.index).rename("d")], axis=1).dropna()
    b = df["a"].rolling(60).cov(df["d"]) / df["d"].rolling(60).var()
    beta_parts[a] = b.reindex(panel.index)
beta_panel = pd.DataFrame(beta_parts, index=panel.index)
sig_dxy = beta_panel.mul(dxy_20.reindex(beta_panel.index), axis=0)

library = {"mom20_volproxy60": sig_mom, "dxy_beta_cond_60x20": sig_dxy}

# ---- Candidate family builders ----
def r_series(s):  # daily simple returns, own calendar
    return s.pct_change()

def rolling_skew(s, w=60):
    r = s.pct_change()
    return r.rolling(w).skew()

def max_drawdown_60(s):
    r = s.pct_change().rolling(60).apply(
        lambda x: float(np.max(np.maximum.accumulate(1 + x.fillna(0)) / (1 + x.fillna(0)) - 1)) if len(x) >= 30 else np.nan,
        raw=True)
    return r

def downside_vol_ratio(s, w=60):
    r = s.pct_change()
    neg = r.where(r < 0, 0.0)
    dsv = np.sqrt((neg ** 2).rolling(w).mean())
    tot = r.rolling(w).std()
    return dsv / tot

def vol_ratio(s, w_long=120, w_short=20):
    r = s.pct_change()
    return r.rolling(w_long).std() / r.rolling(w_short).std()

def gk_vol_ratio(s, w=20):
    # Garman-Klass vol / close-to-close vol over w days
    o = s  # close proxy not needed; use OHLC via separate loader
    return None  # handled separately (needs OHLC)

def autocorr_20(s):
    r = s.pct_change()
    return r.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 10 else np.nan, raw=False)

def reversal_5d_skip2(s):
    return -(s.shift(2) / s.shift(7) - 1.0)

def reversal_10d_volproxy(s):
    # 10d reversal (skip2) damped by 1/(1+|60d mom|) - anti-momentum twin
    rev = -(s.shift(2) / s.shift(12) - 1.0)
    m60 = s / s.shift(60) - 1.0
    return rev / (1.0 + m60.abs())

def amihud_20(s, vol_s):
    r = s.pct_change().abs()
    return r.rolling(20).mean() / vol_s.rolling(20).mean()

def overnight_share(s, w=20):
    # needs open: handled separately
    return None

def usdcny_beta_cond(s, macro_ret, macro_mom, w=60):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), macro_ret.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w).cov(df["m"]) / df["m"].rolling(w).var()
    return b.mul(macro_mom.reindex(b.index), axis=0)

# ---- Build candidate panels ----
cands = {}
cands["downside_vol_ratio_60"] = per_asset(close, downside_vol_ratio, 60)
cands["max_drawdown_60"] = per_asset(close, max_drawdown_60)
cands["skew_60"] = per_asset(close, rolling_skew, 60)
cands["vol_ratio_120_20"] = per_asset(close, vol_ratio, 120, 20)
cands["autocorr_20"] = per_asset(close, autocorr_20)
cands["reversal_5d_skip2"] = per_asset(close, reversal_5d_skip2)
cands["reversal_10d_volproxy"] = per_asset(close, reversal_10d_volproxy)

# USDCNY conditional beta
usdcny = macro_series("USDCNY")
usdcny_ret = usdcny.pct_change()
usdcny_20 = usdcny / usdcny.shift(20) - 1.0
cands["usdcny_beta_cond_60x20"] = per_asset(close, usdcny_beta_cond, usdcny_ret, usdcny_20, 60)

# Amihud with volume (per asset, needs volume panel)
def load_volume_panel():
    frames = {}
    for a in TRADABLES:
        import pandas as pd
        from pathlib import Path
        p = Path("../persistent/stock_data") / f"{a}.csv"
        df = pd.read_csv(p, parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp("2026-07-29")].sort_values("date")
        frames[a] = pd.Series(df["volume"].astype(float).values, index=pd.to_datetime(df["date"]), name=a)
    return pd.concat(frames, axis=1).sort_index()

vol_panel = load_volume_panel()
cands["amihud_20"] = per_asset(close, amihud_20, vol_panel)

# Garman-Klass / OHLC-based candidates
def load_ohlc_panels():
    frames = {}
    for a in TRADABLES:
        import pandas as pd
        from pathlib import Path
        p = Path("../persistent/stock_data") / f"{a}.csv"
        df = pd.read_csv(p, parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp("2026-07-29")].sort_values("date")
        frames[a] = pd.DataFrame({"open": df["open"].values, "high": df["high"].values,
                                  "low": df["low"].values, "close": df["close"].values},
                                 index=pd.to_datetime(df["date"]))
    return frames

ohlc = load_ohlc_panels()

def gk_panel(w=20):
    out = {}
    for a in TRADABLES:
        df = ohlc[a]
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        log_h = np.log(h / o)
        log_l = np.log(l / o)
        log_c = np.log(c / o)
        gk = 0.5 * log_h ** 2 - (2 * np.log(2) - 1) * log_c ** 2 + log_l ** 2
        gk = np.sqrt(gk.rolling(w).mean())
        cc = np.log(c).diff().rolling(w).std()
        out[a] = (gk / cc).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)

def overnight_share_panel(w=20):
    out = {}
    for a in TRADABLES:
        df = ohlc[a]
        o, c = df["open"], df["close"]
        prev_c = c.shift(1)
        on = o / prev_c - 1.0          # overnight return
        intra = c / o - 1.0            # intraday return
        denom = (on + intra).rolling(w).sum()
        share = on.rolling(w).sum() / denom
        out[a] = share.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)

cands["gk_vol_ratio_20"] = gk_panel(20)
cands["overnight_share_20"] = overnight_share_panel(20)

# ---- Validation ----
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)

print("=" * 100)
print("CYCLE 29 EXPLORATION  |  visible through 2026-07-29  |  universe:", len(TRADABLES), "assets")
print("=" * 100)
for name, sig in cands.items():
    m = validate_factor(sig, panel, library=library, fwd_cache=fwd_cache)
    ic, icir = abs(m["ic"]), abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084)
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']} "
          f"turn={m['turnover_10d_rank']} maxlibcorr={m['max_abs_library_correlation']} "
          f"decay={m['decay_ic_by_horizon']} => {'PASS' if passed else 'fail'}")
    print(f"    libcorr: {m.get('library_pairwise_corr')}")
    print()
