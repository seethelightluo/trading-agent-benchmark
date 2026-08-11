"""miner_2: Explore batch B - idiosyncratic / behavioral / liquidity family (2026-07-30).

Motivation: library has raw momentum + trend quality; idiosyncratic (market-residualized)
trend, relative illiquidity, autocorrelation, overnight-move share, market beta and
crypto-comovement are orthogonal families not yet covered.

Candidates:
  idio_trend_20       : 20d asset return minus beta60 * market 20d return (cross-asset EW mkt)
  amihud_rel_60       : Amihud(20) / rolling-60d-mean(Amihud) - 1 (relative illiquidity)
  autocorr_20         : lag-1 return autocorrelation over 20d
  overnight_ratio_20  : mean|open-prev_close| / mean|close-prev_close| (overnight share)
  mkt_beta_60         : rolling beta of asset ret to EW market ret
  crypto_corr_60      : rolling corr of asset ret with BTC ret
  kurt_vol_cond_60x20 : kurtosis_60 * vol_ratio_20x120 (conditional crash-tail)
  vol_revert_20x120   : vol20/vol120 - 1
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

# per-asset clean series helpers
def clean(s):
    return s.dropna()

def roll_apply(s, fn, min_len=30):
    c = clean(s)
    if len(c) < min_len:
        return pd.Series(np.nan, index=idx)
    return fn(c).reindex(idx)

cands = {}

# market = equal-weight cross-asset index
mkt = ret.mean(axis=1)

# --- idio_trend_20: 20d return residualized on market via beta60 ---
def f_idio(s):
    c = clean(ret[s])
    r = c.pct_change()
    pair = pd.concat([r.rename("a"), mkt.rename("m")], axis=1).dropna()
    if len(pair) < 70:
        return pd.Series(np.nan, index=idx)
    beta = pair["a"].rolling(60).cov(pair["m"]) / pair["m"].rolling(60).var().replace(0, np.nan)
    mkt20 = pair["m"].rolling(20).sum()
    r20 = pair["a"].rolling(20).sum()
    idio = (r20 - beta * mkt20).reindex(idx)
    return idio
cands["idio_trend_20"] = pd.DataFrame({s: f_idio(s) for s in WATCH}, index=idx)

# --- amihud_rel_60 ---
vol_data = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    d2 = df.set_index("date")
    vol_data[s] = d2["volume"].astype(float).reindex(idx)

def f_amihud_rel(s):
    c = clean(close[s])
    v = vol_data[s].reindex(c.index).dropna()
    common = c.index.intersection(v.index)
    r = c.reindex(common).pct_change()
    am = (r.abs() / v.reindex(common)).rolling(20).mean()
    base = am.rolling(60).mean()
    out = (am / base.replace(0, np.nan) - 1.0).reindex(idx)
    return out
cands["amihud_rel_60"] = pd.DataFrame({s: f_amihud_rel(s) for s in WATCH}, index=idx)

# --- autocorr_20 ---
def f_ac(s):
    c = clean(ret[s])
    r = c.pct_change()
    def acf(x):
        x = x - x.mean()
        den = (x ** 2).sum()
        if den == 0 or len(x) < 5:
            return np.nan
        return float((x[:-1] * x[1:]).sum() / den)
    return r.rolling(20).apply(acf, raw=True).reindex(idx)
cands["autocorr_20"] = pd.DataFrame({s: f_ac(s) for s in WATCH}, index=idx)

# --- overnight_ratio_20 ---
ohlc = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    d2 = df.set_index("date")
    ohlc[s] = d2[["open", "close"]].reindex(idx)

def f_onight(s):
    d2 = ohlc[s].dropna()
    if len(d2) < 30:
        return pd.Series(np.nan, index=idx)
    prev_close = d2["close"].shift(1)
    gap = (d2["open"] - prev_close).abs()
    tot = (d2["close"] - prev_close).abs()
    ratio = (gap.rolling(20).mean() / tot.rolling(20).mean().replace(0, np.nan)).reindex(idx)
    return ratio
cands["overnight_ratio_20"] = pd.DataFrame({s: f_onight(s) for s in WATCH}, index=idx)

# --- mkt_beta_60 ---
def f_mktbeta(s):
    r = clean(ret[s])
    pair = pd.concat([r.rename("a"), mkt.rename("m")], axis=1).dropna()
    if len(pair) < 70:
        return pd.Series(np.nan, index=idx)
    b = pair["a"].rolling(60).cov(pair["m"]) / pair["m"].rolling(60).var().replace(0, np.nan)
    return b.reindex(idx)
cands["mkt_beta_60"] = pd.DataFrame({s: f_mktbeta(s) for s in WATCH}, index=idx)

# --- crypto_corr_60 ---
btc_ret = clean(ret["BTC"]) if "BTC" in ret.columns else None

def f_cryptocorr(s):
    r = clean(ret[s])
    if btc_ret is None:
        return pd.Series(np.nan, index=idx)
    pair = pd.concat([r.rename("a"), btc_ret.rename("b")], axis=1).dropna()
    if len(pair) < 70:
        return pd.Series(np.nan, index=idx)
    return pair["a"].rolling(60).corr(pair["b"]).reindex(idx)
cands["crypto_corr_60"] = pd.DataFrame({s: f_cryptocorr(s) for s in WATCH}, index=idx)

# --- kurt_vol_cond_60x20 ---
def f_kurtvol(s):
    c = clean(ret[s])
    r = c.pct_change()
    kurt = r.rolling(60).kurt()
    vr = r.rolling(20).std() / r.rolling(120).std().replace(0, np.nan) - 1.0
    return (kurt * vr).reindex(idx)
cands["kurt_vol_cond_60x20"] = pd.DataFrame({s: f_kurtvol(s) for s in WATCH}, index=idx)

# --- vol_revert_20x120 ---
def f_volrev(s):
    c = clean(ret[s])
    r = c.pct_change()
    return (r.rolling(20).std() / r.rolling(120).std().replace(0, np.nan) - 1.0).reindex(idx)
cands["vol_revert_20x120"] = pd.DataFrame({s: f_volrev(s) for s in WATCH}, index=idx)

# --- library IC map ---
def decode_artifact(meta):
    a = meta.get("validation", {}).get("signal_artifact")
    if not a:
        return None
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0, parse_dates=True)
    return sig.reindex(columns=close.columns).reindex(close.index)

lib_ics = {}
for fn in sorted(os.listdir(LIB_DIR)):
    if not fn.endswith(".json") or fn == "factor_ensemble.json":
        continue
    with open(os.path.join(LIB_DIR, fn), encoding="utf-8") as f:
        meta = json.load(f)
    sig = decode_artifact(meta)
    if sig is None:
        continue
    ic = ic_series(sig, fr, min_valid=8)
    if len(ic.dropna()) > 30:
        lib_ics[meta["factor_id"]] = ic
print("library IC series decoded:", len(lib_ics))


def rho_vs_lib(my_ic):
    best, best_id = 0.0, None
    for fid, s in lib_ics.items():
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r) and abs(float(r)) > best:
            best, best_id = abs(float(r)), fid
    return round(best, 4), best_id


print(f"\n--- BATCH B SCREEN (h={H}) ---")
results = {}
for name, sig in cands.items():
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{name:22s} insufficient ({len(ic.dropna())})")
        continue
    m["rho"], m["rho_id"] = rho_vs_lib(ic)
    m["regime"] = regime_split(ic)
    results[name] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    flag = "PASS" if gate else "fail"
    print(f"{name:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} turn={m.get('turnover_10d_rank')} "
          f"rho={m['rho']:.3f}({m['rho_id']}) [{flag}]")
    print("     regime:", json.dumps({k: v["ic"] for k, v in m["regime"].items()}))

with open("scripts/miner2_20260730_explore_batchB_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved -> scripts/miner2_20260730_explore_batchB_results.json")
