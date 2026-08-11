"""miner_3: Explore novel factor families (batch B) - orthogonal signals.

Fixes the batch-A bug (rolling stats were computed on the union calendar with
NaN gaps, wiping out min_periods). Here every rolling statistic is computed on
each asset's OWN clean calendar, then reindexed to the union index.

Families (chosen for orthogonality vs the persisted library of trend/momentum/
macro-beta/vol-of-vol/drawdown-duration):
  Macro FX/rate betas: usdjpy_beta_60, eurusd_beta_60, usdcny_beta_60,
                       xau_beta_60, us10y_beta_20, cn10y_beta_60
  OHLC-shape:          shadow_up_20, shadow_dn_20, shadow_asym_20,
                       high_low_pos_20, overnight_mom_20, intraday_mom_20,
                       range_ratio_5x60
  Return-stat:         pos_days_ratio_20, ret_acorr_1_20, max_dd_20,
                       kurt_20, vol_percentile_250, cs_vol_rank_20
  Volume/flow:         ret_vol_corr_20, volume_surge_5x20, volume_z_20_250
  Correlation:         mkt_corr_20 (corr to equal-weight basket)

Admission gates (15-asset universe): |IC| >= 0.007, |ICIR| >= 0.084,
library IC-series rho < 0.5 preferred. Visible window <= 2026-07-29.
"""
import sys, os, json, io, base64, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"

close = closes_panel(VIS)
idx = close.index
fr = forward_returns(close, H)
ret = close.pct_change()
print(f"visible: {idx.min().date()} .. {idx.max().date()}  n_dates={len(idx)}  n_assets={close.shape[1]}")

# ---- per-asset clean OHLCV frames ----
frames = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].set_index("date").sort_index()
    frames[s] = df

macro = macro_closes(VIS)

# ---- library IC map (artifact decode) ----
def decode_artifact(meta):
    a = meta.get("validation", {}).get("signal_artifact")
    if not a:
        return None
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0, parse_dates=True)
    return sig.reindex(columns=close.columns).reindex(close.index)

def build_lib_ic_map():
    lib = {}
    for fn in sorted(os.listdir(LIB_DIR)):
        if not fn.endswith(".json") or fn == "factor_ensemble.json":
            continue
        with open(os.path.join(LIB_DIR, fn)) as f:
            meta = json.load(f)
        sig = decode_artifact(meta)
        if sig is None:
            continue
        ic = ic_series(sig, fr, min_valid=8)
        if len(ic.dropna()) > 30:
            lib[meta["factor_id"]] = ic
    return lib

lib_ics = build_lib_ic_map()
print("library IC series decoded:", len(lib_ics), sorted(lib_ics.keys()))

def rho_vs_lib(my_ic):
    best = 0.0
    for fid, s in lib_ics.items():
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r):
            best = max(best, abs(float(r)))
    return round(best, 4)

cands = {}

# ---- 1) macro FX/rate betas ----
def macro_beta_family(macro_name):
    m = macro[macro_name]
    out = {}
    for s in WATCH:
        c = frames[s]["close"]
        r = c.pct_change()
        mm = m.reindex(c.index)
        pair = pd.concat([r.rename("a"), mm.rename("m")], axis=1).dropna()
        if len(pair) < 70:
            out[s] = pd.Series(np.nan, index=idx)
            continue
        b = pair["a"].rolling(60).cov(pair["m"]) / pair["m"].rolling(60).var()
        out[s] = b.reindex(idx)
    return pd.DataFrame(out, index=idx)

for mn in ["USDJPY", "EURUSD", "USDCNY"]:
    cands[f"{mn.lower()}_beta_60"] = macro_beta_family(mn)

def asset_beta_family(asset, win):
    a_ret = frames[asset]["close"].pct_change()
    out = {}
    for s in WATCH:
        c = frames[s]["close"]
        r = c.pct_change()
        aa = a_ret.reindex(c.index)
        pair = pd.concat([r.rename("a"), aa.rename("m")], axis=1).dropna()
        if len(pair) < win + 10:
            out[s] = pd.Series(np.nan, index=idx)
            continue
        b = pair["a"].rolling(win).cov(pair["m"]) / pair["m"].rolling(win).var()
        out[s] = b.reindex(idx)
    return pd.DataFrame(out, index=idx)

cands["xau_beta_60"] = asset_beta_family("XAU", 60)
cands["us10y_beta_20"] = asset_beta_family("US10Y", 20)
cands["cn10y_beta_60"] = asset_beta_family("CN10Y", 60)

# ---- 2) OHLC-shape (clean per-asset) ----
def ohlc_series(s, col):
    return frames[s][col]

def _shadow(s, kind):
    d = frames[s]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    if kind == "up":
        x = (d["high"] - np.maximum(d["open"], d["close"])) / rng
    elif kind == "dn":
        x = (np.minimum(d["open"], d["close"]) - d["low"]) / rng
    else:
        up = (d["high"] - np.maximum(d["open"], d["close"])) / rng
        dn = (np.minimum(d["open"], d["close"]) - d["low"]) / rng
        x = up - dn
    return x.rolling(20).mean().reindex(idx)

cands["shadow_up_20"] = pd.DataFrame({s: _shadow(s, "up") for s in WATCH}, index=idx)
cands["shadow_dn_20"] = pd.DataFrame({s: _shadow(s, "dn") for s in WATCH}, index=idx)
cands["shadow_asym_20"] = pd.DataFrame({s: _shadow(s, "asym") for s in WATCH}, index=idx)

def _hlp(s):
    d = frames[s]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    return ((d["close"] - d["low"]) / rng).rolling(20).mean().reindex(idx)
cands["high_low_pos_20"] = pd.DataFrame({s: _hlp(s) for s in WATCH}, index=idx)

def _overnight(s):
    d = frames[s]
    return (d["open"] / d["close"].shift(1) - 1.0).rolling(20).mean().reindex(idx)
cands["overnight_mom_20"] = pd.DataFrame({s: _overnight(s) for s in WATCH}, index=idx)

def _intraday(s):
    d = frames[s]
    return (d["close"] / d["open"] - 1.0).rolling(20).mean().reindex(idx)
cands["intraday_mom_20"] = pd.DataFrame({s: _intraday(s) for s in WATCH}, index=idx)

def _rr(s):
    d = frames[s]
    tr = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    a5 = tr.rolling(5).mean()
    a60 = tr.rolling(60).mean().replace(0, np.nan)
    return (a5 / a60).reindex(idx)
cands["range_ratio_5x60"] = pd.DataFrame({s: _rr(s) for s in WATCH}, index=idx)

# ---- 3) return-stat ----
def _posdays(s):
    c = frames[s]["close"]
    return (c.pct_change() > 0).astype(float).rolling(20).mean().reindex(idx)
cands["pos_days_ratio_20"] = pd.DataFrame({s: _posdays(s) for s in WATCH}, index=idx)

def _acorr1(s):
    c = frames[s]["close"]
    r = c.pct_change()
    return r.rolling(20).corr(r.shift(1)).reindex(idx)
cands["ret_acorr_1_20"] = pd.DataFrame({s: _acorr1(s) for s in WATCH}, index=idx)

def _maxdd20(s):
    c = frames[s]["close"]
    return (c / c.rolling(20).max() - 1.0).reindex(idx)
cands["max_dd_20"] = pd.DataFrame({s: _maxdd20(s) for s in WATCH}, index=idx)

def _kurt(s):
    c = frames[s]["close"]
    return c.pct_change().rolling(20).kurt().reindex(idx)
cands["kurt_20"] = pd.DataFrame({s: _kurt(s) for s in WATCH}, index=idx)

def _volpct(s):
    c = frames[s]["close"]
    r = c.pct_change()
    v20 = r.rolling(20).std()
    mu = r.rolling(250).mean()
    sd = r.rolling(250).std().replace(0, np.nan)
    return ((v20 - mu) / sd).reindex(idx)
cands["vol_percentile_250"] = pd.DataFrame({s: _volpct(s) for s in WATCH}, index=idx)

# cross-sectional rank of 20d vol (within date)
v20 = pd.DataFrame({s: frames[s]["close"].pct_change().rolling(20).std().reindex(idx) for s in WATCH}, index=idx)
cands["cs_vol_rank_20"] = v20.rank(axis=1, pct=True)

# ---- 4) volume/flow (assets with positive volume only) ----
def _rvc(s):
    d = frames[s]
    if (d["volume"].fillna(0) <= 0).all():
        return pd.Series(np.nan, index=idx)
    r = d["close"].pct_change()
    vr = d["volume"].pct_change()
    pair = pd.concat([r.rename("r"), vr.rename("v")], axis=1).dropna()
    if len(pair) < 30:
        return pd.Series(np.nan, index=idx)
    return pair["r"].rolling(20, min_periods=10).corr(pair["v"]).reindex(idx)
cands["ret_vol_corr_20"] = pd.DataFrame({s: _rvc(s) for s in WATCH}, index=idx)

def _volsurge(s):
    d = frames[s]
    if (d["volume"].fillna(0) <= 0).all():
        return pd.Series(np.nan, index=idx)
    v = d["volume"]
    return (v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)).reindex(idx)
cands["volume_surge_5x20"] = pd.DataFrame({s: _volsurge(s) for s in WATCH}, index=idx)

def _volz(s):
    d = frames[s]
    if (d["volume"].fillna(0) <= 0).all():
        return pd.Series(np.nan, index=idx)
    v = d["volume"]
    m20 = v.rolling(20).mean()
    mu = v.rolling(250).mean()
    sd = v.rolling(250).std().replace(0, np.nan)
    return ((m20 - mu) / sd).reindex(idx)
cands["volume_z_20_250"] = pd.DataFrame({s: _volz(s) for s in WATCH}, index=idx)

# ---- 5) correlation to basket ----
basket = ret.mean(axis=1, skipna=True)
def _mktcorr(s):
    c = frames[s]["close"]
    r = c.pct_change()
    b = basket.reindex(c.index)
    pair = pd.concat([r.rename("a"), b.rename("m")], axis=1).dropna()
    if len(pair) < 30:
        return pd.Series(np.nan, index=idx)
    return pair["a"].rolling(20, min_periods=10).corr(pair["m"]).reindex(idx)
cands["mkt_corr_20"] = pd.DataFrame({s: _mktcorr(s) for s in WATCH}, index=idx)

# ---- screen ----
print("\n--- CANDIDATE SCREEN (h=%d, min_valid=8) ---" % H)
results = {}
for name, sig in cands.items():
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{name:22s} insufficient dates ({len(ic.dropna())})")
        continue
    m["rho"] = rho_vs_lib(ic)
    m["regime"] = regime_split(ic)
    results[name] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    rho_ok = m["rho"] < 0.5
    flag = "PASS" if (gate and rho_ok) else ("gate-ok-rho-hi" if gate else "fail")
    print(f"{name:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} turn={m['turnover_10d_rank']:.3f} "
          f"rho={m['rho']:.3f} [{flag}]")
    print("     regime:", json.dumps(m["regime"]))

with open("scripts/miner3_20260730_explore_batchB_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved -> scripts/miner3_20260730_explore_batchB_results.json")
