"""miner_3: Explore novel factor families (batch A) - risk, OHLC-shape, and
return-quality signals on the 15-asset cross-asset universe.

Ideas (all distinct from the persisted library of trend/momentum/macro-beta/
vol-of-vol/drawdown-duration factors):
  1. mkt_beta_60        : 60d beta of asset returns to equal-weight 15-asset basket (systematic risk)
  2. idio_vol_ratio_20  : residual (idiosyncratic) vol after basket-beta / total vol (20d)
  3. shadow_up_20       : mean upper candlestick shadow ratio over 20d (selling pressure)
  4. shadow_dn_20       : mean lower candlestick shadow ratio over 20d (buying support)
  5. shadow_asym_20     : upper - lower shadow asymmetry over 20d
  6. updown_capture_20  : mean up-day return / |mean down-day return| (return quality)
  7. max_ret_20         : max daily return over 20d (upside tail)
  8. min_ret_20         : min daily return over 20d (downside tail)
  9. range_ratio_5x60   : 5d avg daily range / 60d avg daily range (range contraction/expansion)
 10. ret_vol_corr_20    : rolling corr of daily return with volume change (20d)
 11. losing_streak_60   : max consecutive losing days within trailing 60d
 12. overnight_mom_20   : mean overnight return (open/prev_close - 1) over 20d
 13. intraday_mom_20    : mean intraday return (close/open - 1) over 20d
 14. high_low_pos_20    : mean close position inside daily high-low range over 20d
 15. eff_ratio_20       : Kaufman efficiency ratio 20d (net path / gross path)

Only the visible window (<= 2026-07-29) is used. Cross-sectional rank IC vs 10d
forward returns, min 8 valid instruments per date.
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
ret = close.pct_change()
n_dates = len(close)
print(f"visible window: {idx.min().date()} .. {idx.max().date()}  n_dates={n_dates}  n_assets={close.shape[1]}")

# ---- library IC map by decoding artifacts (for rho) ----
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
print("library IC series decoded:", len(lib_ics), list(lib_ics.keys()))

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

# ---- per-asset clean-series helpers ----
def clean(s):
    return s.dropna()

def reidx(s):
    return s.reindex(idx)

# ---- OHLC + volume data (visible window) ----
ohlc = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)]
    d2 = df.set_index("date")
    ohlc[s] = d2[["open", "high", "low", "close", "volume"]].astype(float).reindex(idx)

cands = {}

# 1) mkt_beta_60: beta to equal-weight basket return
basket = ret.mean(axis=1, skipna=True)
def _mkt_beta(s):
    c = clean(close[s])
    r = c.pct_change()
    b = basket.reindex(c.index)
    pair = pd.concat([r.rename("a"), b.rename("m")], axis=1).dropna()
    if len(pair) < 70:
        return pd.Series(np.nan, index=idx)
    beta = pair["a"].rolling(60).cov(pair["m"]) / pair["m"].rolling(60).var()
    return beta.reindex(idx)
cands["mkt_beta_60"] = pd.DataFrame({s: _mkt_beta(s) for s in WATCH}, index=idx)

# 2) idio_vol_ratio_20: residual vol after basket beta / total vol (20d)
def _idio(s):
    c = clean(close[s])
    r = c.pct_change()
    b = basket.reindex(c.index)
    pair = pd.concat([r.rename("a"), b.rename("m")], axis=1).dropna()
    if len(pair) < 70:
        return pd.Series(np.nan, index=idx)
    beta = pair["a"].rolling(60).cov(pair["m"]) / pair["m"].rolling(60).var()
    resid = pair["a"] - beta * pair["m"]
    idio = resid.rolling(20).std()
    tot = pair["a"].rolling(20).std()
    return (idio / tot.replace(0, np.nan)).reindex(idx)
cands["idio_vol_ratio_20"] = pd.DataFrame({s: _idio(s) for s in WATCH}, index=idx)

# 3-5) candlestick shadow ratios
def _shadow(s, kind):
    d = ohlc[s]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    if kind == "up":
        x = (d["high"] - np.maximum(d["open"], d["close"])) / rng
    elif kind == "dn":
        x = (np.minimum(d["open"], d["close"]) - d["low"]) / rng
    else:  # asym
        up = (d["high"] - np.maximum(d["open"], d["close"])) / rng
        dn = (np.minimum(d["open"], d["close"]) - d["low"]) / rng
        x = up - dn
    return x.rolling(20).mean()
cands["shadow_up_20"] = pd.DataFrame({s: _shadow(s, "up") for s in WATCH}, index=idx)
cands["shadow_dn_20"] = pd.DataFrame({s: _shadow(s, "dn") for s in WATCH}, index=idx)
cands["shadow_asym_20"] = pd.DataFrame({s: _shadow(s, "asym") for s in WATCH}, index=idx)

# 6) updown_capture_20
def _udc(s):
    c = clean(close[s])
    r = c.pct_change()
    up = r.clip(lower=0.0).rolling(20).mean()
    dn = (-r.clip(upper=0.0)).rolling(20).mean().replace(0, np.nan)
    return (up / dn).reindex(idx)
cands["updown_capture_20"] = pd.DataFrame({s: _udc(s) for s in WATCH}, index=idx)

# 7-8) max/min daily return over 20d
cands["max_ret_20"] = pd.DataFrame({s: clean(close[s]).pct_change().rolling(20).max().reindex(idx) for s in WATCH}, index=idx)
cands["min_ret_20"] = pd.DataFrame({s: clean(close[s]).pct_change().rolling(20).min().reindex(idx) for s in WATCH}, index=idx)

# 9) range_ratio_5x60
def _rr(s):
    d = ohlc[s]
    tr = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    a5 = tr.rolling(5).mean()
    a60 = tr.rolling(60).mean().replace(0, np.nan)
    return (a5 / a60).reindex(idx)
cands["range_ratio_5x60"] = pd.DataFrame({s: _rr(s) for s in WATCH}, index=idx)

# 10) ret_vol_corr_20
def _rvc(s):
    d = ohlc[s]
    r = d["close"].pct_change()
    vr = d["volume"].pct_change()
    pair = pd.concat([r.rename("r"), vr.rename("v")], axis=1).dropna()
    if len(pair) < 30:
        return pd.Series(np.nan, index=idx)
    return pair["r"].rolling(20, min_periods=10).corr(pair["v"]).reindex(idx)
cands["ret_vol_corr_20"] = pd.DataFrame({s: _rvc(s) for s in WATCH}, index=idx)

# 11) losing_streak_60
def _ls(s):
    c = clean(close[s])
    r = c.pct_change()
    neg = (r < 0).astype(int)
    grp = (neg == 0).cumsum()
    streak = neg.groupby(grp).cumsum()
    return streak.rolling(60, min_periods=30).max().reindex(idx)
cands["losing_streak_60"] = pd.DataFrame({s: _ls(s) for s in WATCH}, index=idx)

# 12-13) overnight / intraday momentum
def _overnight(s):
    d = ohlc[s]
    return (d["open"] / d["close"].shift(1) - 1.0).rolling(20).mean().reindex(idx)
cands["overnight_mom_20"] = pd.DataFrame({s: _overnight(s) for s in WATCH}, index=idx)

def _intraday(s):
    d = ohlc[s]
    return (d["close"] / d["open"] - 1.0).rolling(20).mean().reindex(idx)
cands["intraday_mom_20"] = pd.DataFrame({s: _intraday(s) for s in WATCH}, index=idx)

# 14) high_low_pos_20
def _hlp(s):
    d = ohlc[s]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    return ((d["close"] - d["low"]) / rng).rolling(20).mean().reindex(idx)
cands["high_low_pos_20"] = pd.DataFrame({s: _hlp(s) for s in WATCH}, index=idx)

# 15) eff_ratio_20
def _eff(s):
    c = clean(close[s])
    r = c.pct_change().abs()
    net = (c - c.shift(20)).abs()
    gross = r.rolling(20).sum().replace(0, np.nan)
    return (net / gross).reindex(idx)
cands["eff_ratio_20"] = pd.DataFrame({s: _eff(s) for s in WATCH}, index=idx)

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

with open("scripts/miner3_20260730_explore_batchA_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved -> scripts/miner3_20260730_explore_batchA_results.json")
