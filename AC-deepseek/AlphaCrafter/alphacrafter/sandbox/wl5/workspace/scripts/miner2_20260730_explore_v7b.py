"""miner_2: Explore new factor families v2 (rolling stats on per-asset calendars).

Key fix: rolling/shift stats computed on each asset's own clean series, then
reindexed to the union calendar. Library IC series decoded from artifacts for
accurate rho.
"""
import sys, os, json, io, base64, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"

close = closes_panel(VIS)
idx = close.index
fr = forward_returns(close, H)

# ---- library IC map by decoding artifacts ----
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
print("library IC series decoded:", len(lib_ics))


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


# ---- per-asset clean-series computation ----
def clean(s):
    return s.dropna()


def roll(s, fn, *args):
    """Apply rolling fn on clean series, return reindexed to union index."""
    c = clean(s)
    if len(c) < 30:
        return pd.Series(np.nan, index=idx)
    return fn(c, *args).reindex(idx)


def roll_close(s, fn, *args):
    c = clean(s)
    if len(c) < 30:
        return pd.Series(np.nan, index=idx)
    return fn(c, *args).reindex(idx)


# ---- data for volume/high-low ----
vol_data = {}
hl_data = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    d2 = df.set_index("date")
    vol_data[s] = d2["volume"].astype(float).reindex(idx)
    hl_data[s] = d2[["high", "low"]].reindex(idx)

ret = close.pct_change()

cands = {}

# drawdown depth vs rolling max (60/120)
for w in (60, 120):
    def _dd(c, w=w):
        return c / c.rolling(w).max() - 1.0
    cands[f"dd_{w}"] = pd.DataFrame({s: roll(close[s], _dd) for s in WATCH}, index=idx)

# time under water (days since last rolling-max, logged)
def _tu(c):
    rmax = c.rolling(120).max()
    tu = (c.rolling(120).apply(lambda x: len(x) - 1 - int(np.argmax(x.values)), raw=True))
    return tu
cands["time_under_water_120"] = pd.DataFrame({s: roll(close[s], _tu) for s in WATCH}, index=idx)

# RSI(14)
def _rsi(c):
    delta = c.diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    dn = (-delta.clip(upper=0.0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)
cands["rsi_14"] = pd.DataFrame({s: roll(close[s], _rsi) for s in WATCH}, index=idx)

# Amihud illiquidity 20d
def _amihud(s):
    c = clean(close[s])
    v = vol_data[s].reindex(c.index).dropna()
    common = c.index.intersection(v.index)
    r = c.reindex(common).pct_change()
    return (r.abs() / v.reindex(common)).rolling(20).mean().reindex(idx)
cands["amihud_20"] = pd.DataFrame({s: _amihud(s) for s in WATCH}, index=idx)

# volume trend 20x60
def _vt(s):
    v = vol_data[s].dropna()
    if len(v) < 70:
        return pd.Series(np.nan, index=idx)
    return (v.rolling(20).mean() / v.rolling(60).mean() - 1.0).reindex(idx)
cands["vol_trend_20x60"] = pd.DataFrame({s: _vt(s) for s in WATCH}, index=idx)

# downside/upside semi-vol asymmetry
def _semi(s):
    c = clean(close[s])
    r = c.pct_change()
    neg = r.clip(upper=0.0)
    pos = r.clip(lower=0.0)
    d = (neg ** 2).rolling(20).mean().apply(np.sqrt)
    u = (pos ** 2).rolling(20).mean().apply(np.sqrt)
    return (d / u.replace(0, np.nan) - 1.0).reindex(idx)
cands["semi_down_ratio_20"] = pd.DataFrame({s: _semi(s) for s in WATCH}, index=idx)

# 20d range position
def _rp(s):
    c = clean(close[s])
    h = hl_data[s]["high"].reindex(c.index)
    l = hl_data[s]["low"].reindex(c.index)
    d2 = pd.concat([c.rename("c"), h.rename("h"), l.rename("l")], axis=1).dropna()
    if len(d2) < 30:
        return pd.Series(np.nan, index=idx)
    hi = d2["h"].rolling(20).max()
    lo = d2["l"].rolling(20).min()
    rng = (hi - lo).replace(0, np.nan)
    return ((d2["c"] - lo) / rng).reindex(idx)
cands["rng_pos_20"] = pd.DataFrame({s: _rp(s) for s in WATCH}, index=idx)

# return/vol ratio 20d (risk-adjusted trend)
def _rv(s):
    c = clean(close[s])
    r = c.pct_change()
    return (r.rolling(20).mean() / r.rolling(20).std().replace(0, np.nan)).reindex(idx)
cands["ret_vol_ratio_20"] = pd.DataFrame({s: _rv(s) for s in WATCH}, index=idx)

# raw 20d skewness
def _sk(s):
    c = clean(close[s])
    return c.pct_change().rolling(20).skew().reindex(idx)
cands["skew_20_raw"] = pd.DataFrame({s: _sk(s) for s in WATCH}, index=idx)

# 60d kurtosis
def _ku(s):
    c = clean(close[s])
    r = c.pct_change()
    return r.rolling(60).kurt().reindex(idx)
cands["kurt_60"] = pd.DataFrame({s: _ku(s) for s in WATCH}, index=idx)

# vol ratio 20x60 (short/long vol)
def _vr(s):
    c = clean(close[s])
    r = c.pct_change()
    return (r.rolling(20).std() / r.rolling(60).std().replace(0, np.nan) - 1.0).reindex(idx)
cands["vol_ratio_20x60"] = pd.DataFrame({s: _vr(s) for s in WATCH}, index=idx)

# 60d momentum skip5 (redundancy reference)
cands["mom_60_skip5"] = close.shift(5) / close.shift(65) - 1.0

print("\n--- CANDIDATE SCREEN (h=%d) ---" % H)
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
    flag = "PASS" if gate else "fail"
    print(f"{name:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} rho={m['rho']:.3f} [{flag}]")
    print("     regime:", json.dumps(m["regime"]))

with open("scripts/miner2_20260730_explore_v7b_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved results -> scripts/miner2_20260730_explore_v7b_results.json")
