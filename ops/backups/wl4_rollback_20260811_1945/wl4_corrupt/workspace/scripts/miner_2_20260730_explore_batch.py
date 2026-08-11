"""miner_2 exploration batch 2026-07-30 (optimized).
Screen candidate factors on the 15-asset tradable cross-asset universe.
Validation window: 2020-01-01 .. 2026-07-29. Admission horizon H=10.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import numpy as np
import pandas as pd

TRADING_END = "2026-07-29"
START = "2020-01-01"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]

closes, vols, highs, lows = {}, {}, {}, {}
for a in ASSETS:
    d = load(f"../persistent/stock_data/{a}.csv")
    closes[a] = d["close"].astype(float); vols[a] = d["volume"].astype(float)
    highs[a] = d["high"].astype(float); lows[a] = d["low"].astype(float)
close = pd.DataFrame(closes); vol = pd.DataFrame(vols)
high = pd.DataFrame(highs); low = pd.DataFrame(lows)
mac = {m: load(f"../persistent/index_data/{m}.csv")["close"].astype(float) for m in MACRO}
macro = pd.DataFrame(mac)

close = close.loc[(close.index >= START) & (close.index <= TRADING_END)]
vol = vol.reindex(close.index); high = high.reindex(close.index); low = low.reindex(close.index)
macro = macro.reindex(close.index)
ret = close.pct_change()
print(f"panel dates: {len(close)}  assets: {close.shape[1]}  {close.index[0].date()}..{close.index[-1].date()}")

def fwd(h):
    return close.shift(-h) / close - 1.0

def rolling_beta(asset_ret, factor_ret, win):
    # E[xy]-E[x]E[y] over rolling window; factor_ret aligned Series
    exy = asset_ret.mul(factor_ret, axis=0).rolling(win).mean()
    ex = asset_ret.rolling(win).mean()
    ey = factor_ret.rolling(win).mean()
    var = factor_ret.rolling(win).var()
    cov = exy - ex.mul(ey, axis=0)
    return cov.div(var, axis=0)

def zscore_rows(panel):
    m = panel.values.astype(float)
    mu = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, keepdims=True)
    sd[sd < 1e-12] = np.nan
    return (m - mu) / sd

def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1.0, len(x) + 1.0)
    return r

def daily_ic_np(factor_panel, fwd_panel, min_valid=8):
    F = factor_panel.values.astype(float)
    R = fwd_panel.values.astype(float)
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        f, r = F[i], R[i]
        m = np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n >= min_valid and n >= 2:
            a, b = _rank(f[m]), _rank(r[m])
            c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c):
                out[i] = c
    return out

def library_corr_np(factor_panel, lib, min_overlap=100):
    zf = zscore_rows(factor_panel)
    out = {}
    for name, s in lib.items():
        s = s.reindex(factor_panel.index)
        zs = zscore_rows(s)
        mask = np.isfinite(zf) & np.isfinite(zs)
        cnt = int(mask.sum())
        if cnt > min_overlap:
            c = np.corrcoef(zf[mask], zs[mask])[0, 1]
            out[name] = float(c)
    return out

cand = {}
# 1 efficiency ratio 20d
cand["eff_ratio_20"] = (close - close.shift(20)).abs() / ret.abs().rolling(20).sum()
# 2 position within 20d range
hh20, ll20 = high.rolling(20).max(), low.rolling(20).min()
cand["range_pos_20"] = (close - ll20) / (hh20 - ll20)
# 3 20d range width
cand["range_w_20"] = (hh20 - ll20) / close
# 4 skewness 60d
cand["skew_60"] = ret.rolling(60).skew()
# 5 downside concentration
neg = ret.clip(upper=0.0)
cand["down_ratio_20x60"] = neg.pow(2).rolling(20).mean().pow(0.5) / ret.rolling(60).std()
# 6 drawdown depth 60d
cand["drawdown_60"] = close / close.rolling(60).max() - 1.0
# 7 volume trend 20x60
cand["vol_trend_20x60"] = vol.rolling(20).mean() / vol.rolling(60).mean()
# 8 momentum acceleration
cand["mom_accel_10x60"] = (close / close.shift(10) - 1.0) - (close / close.shift(60) - 1.0)
# 9 RSI 14
d = close.diff(); up = d.clip(lower=0.0).rolling(14).mean(); dn = (-d.clip(upper=0.0)).rolling(14).mean()
cand["rsi_14"] = 100.0 - 100.0 / (1.0 + up / dn)
# 10 DXY beta conditional
cand["dxy_beta_cond_60x20"] = rolling_beta(ret, macro["DXY"].pct_change(), 60).mul(macro["DXY"] / macro["DXY"].shift(20) - 1.0, axis=0)
# 11 US10Y beta conditional
cand["us10y_beta_cond_60x20"] = rolling_beta(ret, close["US10Y"].pct_change(), 60).mul(close["US10Y"] / close["US10Y"].shift(20) - 1.0, axis=0)
# 12 USDCNY beta conditional
cand["cny_beta_cond_60x20"] = rolling_beta(ret, macro["USDCNY"].pct_change(), 60).mul(macro["USDCNY"] / macro["USDCNY"].shift(20) - 1.0, axis=0)
# 13 XAU beta conditional
cand["xau_beta_cond_60x20"] = rolling_beta(ret, close["XAU"].pct_change(), 60).mul(close["XAU"] / close["XAU"].shift(20) - 1.0, axis=0)
# 14 BTC beta conditional
cand["btc_beta_cond_60x20"] = rolling_beta(ret, close["BTC"].pct_change(), 60).mul(close["BTC"] / close["BTC"].shift(20) - 1.0, axis=0)
# 15 10d avg intraday range
cand["hl_range_10"] = ((high - low) / close).rolling(10).mean()

# library existing effective factors
lib = {}
lib["mom_10d_skip5"] = close.shift(5) / close.shift(15) - 1.0
lib["mom_120d_skip5"] = close.shift(5) / close.shift(125) - 1.0
vix_ret = macro["VIX"].pct_change()
lib["vix_beta_cond_60x20"] = -rolling_beta(ret, vix_ret, 60).mul(macro["VIX"] / macro["VIX"].shift(20) - 1.0, axis=0)
lib["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()

horizons = [1, 2, 3, 5, 10, 20]
fwd_panels = {h: fwd(h) for h in horizons}
fwd10 = fwd_panels[10]

results = []
for name, s in cand.items():
    s = s.reindex(close.index)
    ic_arr = daily_ic_np(s, fwd10)
    ic = float(np.nanmean(ic_arr)); icsd = float(np.nanstd(ic_arr))
    icir = ic / icsd if icsd > 0 else 0.0
    hit = float(np.nanmean(ic_arr > 0)) if len(ic_arr) else float("nan")
    cov = float(s.notna().sum().sum()) / float(s.size)
    ranks = s.rank(axis=1)
    to = float(ranks.diff(10).abs().mean(axis=1).dropna().mean())
    lc = library_corr_np(s, lib)
    maxlc = max(abs(v) for v in lc.values()) if lc else 0.0
    dec = {str(h): round(float(np.nanmean(daily_ic_np(s, fwd_panels[h]))), 4) for h in horizons}
    results.append(dict(name=name, ic=ic, icir=icir, hit=hit, cov=cov, to=to,
                        maxlc=maxlc, n_ic=int(np.isfinite(ic_arr).sum()), decay=dec, lib_corr=lc))

res = pd.DataFrame(results).sort_values("icir", key=lambda x: x.abs(), ascending=False)
pd.set_option("display.width", 250)
print(res[["name", "ic", "icir", "hit", "cov", "to", "maxlc", "n_ic"]].to_string(index=False))
print("\nDecay IC by horizon:")
for _, r in res.iterrows():
    print(f"  {r['name']:<24} " + " ".join(f"{k}:{v:+.4f}" for k, v in r["decay"].items()))
print("\nGate check |IC|>=0.007 & |ICIR|>=0.084:")
for _, r in res.iterrows():
    ok = abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084
    print(f"  {'PASS' if ok else 'fail'}  {r['name']:<24} ic={r['ic']:+.4f} icir={r['icir']:+.4f}")
print("\nLibrary correlations (top candidates):")
for _, r in res.head(8).iterrows():
    print(f"  {r['name']:<24} " + " ".join(f"{k}:{v:+.3f}" for k, v in r["lib_corr"].items()))
