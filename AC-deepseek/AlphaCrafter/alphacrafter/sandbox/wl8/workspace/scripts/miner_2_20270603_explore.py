"""miner_2 2027-06-03: explore fresh candidate factors on the 15-asset cross-asset
universe. Data visible through 2027-06-02 (previous completed trading day).

Batch A - cross-asset beta / macro linkage:
  xau_beta_60            : 60d beta(ret_asset, ret_XAU)            (gold beta)
  wti_beta_60            : 60d beta(ret_asset, ret_WTI)            (oil beta)
  btc_beta_60            : 60d beta(ret_asset, ret_BTC)            (crypto beta)
  us10y_beta_60          : 60d beta(ret_asset, ret_US10Y)          (rate beta, returns)
  yield_spread_beta_60   : 60d beta(ret_asset, ret(US10Y-CN10Y))   (spread beta)

Batch B - volatility / risk:
  vol_z_20               : (vol20 - mean(vol20,60))/std(vol20,60)  (vol z-score)
  downside_vol_20        : std of negative daily returns (20d)
  skew_60                : 60d return skewness
  max_gain_20            : max 1d return over 20d (lottery demand)
  vol_ratio_5x20         : vol5 / vol20

Batch C - trend / oscillator:
  rsi_5                  : 5-day RSI (short-term mean reversion)
  trend_strength_20      : |mom20| / sum(|ret|)  (efficiency ratio 20d)
  mom_5d_skip2           : 5d momentum skip 2
  hl_pos_10              : (close-min10)/(max10-min10)

Batch D - liquidity / volume:
  volume_trend_20x60     : avg volume 20d / avg volume 60d
  pv_corr_20             : 20d corr(daily price chg, volume)

Admission gates: |IC10|>=0.0070 and |ICIR10|>=0.0840; |rho| vs active library
(usdcny_beta_60) < 0.5. Also re-validates usdcny_beta_60 for drift monitoring.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

IC_THR = 0.0070
ICIR_THR = 0.0840
RHO_THR = 0.5
MIN_ASSETS = 8
MIN_IC_DATES = 60

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
INDEX_DATA_DIR = "../persistent/index_data/"


def load_asset(symbol, days=3200):
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            df = get_stock_daily_data(symbol=symbol, days=days)
        except Exception:
            df = None
    if df is None or len(df) < 400:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


print("loading data ...")
data = {a: load_asset(a, days=3200) for a in WATCH}
data = {a: d for a, d in data.items() if d is not None}
closes = {a: d["close"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float) if "volume" in d.columns and d["volume"].notna().any() else None
        for a, d in data.items()}
last_vis = max(d.index.max() for d in data.values())
print(f"loaded {len(data)}/15 instruments; last_vis={last_vis.date()}")

# macro observation series truncated to visible window
def load_obs(name):
    df = pd.read_csv(f"{INDEX_DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").sort_index()["close"].astype(float)
    return s[s.index <= last_vis]

OBS = {m: load_obs(m) for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
print(f"macro last dates: { {m: s.index.max().date() for m, s in OBS.items()} }")

fclose = pd.DataFrame(closes).sort_index()
fvol = pd.DataFrame({a: v for a, v in vols.items() if v is not None}).sort_index()
print(f"volume panel: {fvol.shape} (assets with volume: {fvol.shape[1]})")

fwd10 = fclose.shift(-10) / fclose - 1.0
fwd1 = fclose.shift(-1) / fclose - 1.0
fwd3 = fclose.shift(-3) / fclose - 1.0
fwd5 = fclose.shift(-5) / fclose - 1.0
fwd20 = fclose.shift(-20) / fclose - 1.0
rets = fclose.pct_change()


def ic_series(factor, fwd, min_assets=MIN_ASSETS):
    ics, dates = [], []
    for dt in factor.index.intersection(fwd.index):
        f = factor.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        both = f.index.intersection(r.index)
        if len(both) < min_assets:
            continue
        x, y = f[both], r[both]
        if x.nunique() < 3 or y.nunique() < 3:
            continue
        rho = spearmanr(x, y)[0]
        if np.isfinite(rho):
            ics.append(rho)
            dates.append(dt)
    return pd.Series(ics, index=dates)


def metrics(ics):
    n = len(ics)
    if n < MIN_IC_DATES:
        return dict(ic=np.nan, icir=np.nan, n=n, hit=np.nan)
    ic = float(ics.mean())
    sd = float(ics.std(ddof=1)) if n > 1 else 0.0
    icir = ic / sd if sd > 0 else np.nan
    hit = float((ics > 0).mean())
    return dict(ic=ic, icir=icir, n=n, hit=hit)


def build_all():
    F = {}
    rspx = rets["SPX"]
    rxau = rets["XAU"]
    rwti = rets["WTI"]
    rbtc = rets["BTC"]
    rus10 = rets["US10Y"]
    rcn10 = rets["CN10Y"]
    spread = (closes["US10Y"] - closes["CN10Y"]).reindex(fclose.index)
    rspread = spread.pct_change().replace([np.inf, -np.inf], np.nan)

    for a, c in closes.items():
        r = c.pct_change()
        # ---- Batch A: cross-asset beta ----
        F.setdefault("xau_beta_60", {})[a] = r.rolling(60).cov(rxau) / rxau.rolling(60).var()
        F.setdefault("wti_beta_60", {})[a] = r.rolling(60).cov(rwti) / rwti.rolling(60).var()
        F.setdefault("btc_beta_60", {})[a] = r.rolling(60).cov(rbtc) / rbtc.rolling(60).var()
        F.setdefault("us10y_beta_60", {})[a] = r.rolling(60).cov(rus10) / rus10.rolling(60).var()
        F.setdefault("yield_spread_beta_60", {})[a] = r.rolling(60).cov(rspread) / rspread.rolling(60).var()
        # ---- Batch B: vol / risk ----
        v20 = r.rolling(20).std()
        m20 = v20.rolling(60).mean()
        s20 = v20.rolling(60).std()
        F.setdefault("vol_z_20", {})[a] = (v20 - m20) / s20.replace(0, np.nan)
        neg = r.where(r < 0)
        F.setdefault("downside_vol_20", {})[a] = neg.rolling(20).std()
        F.setdefault("skew_60", {})[a] = r.rolling(60).skew()
        F.setdefault("max_gain_20", {})[a] = r.rolling(20).max()
        v5 = r.rolling(5).std()
        F.setdefault("vol_ratio_5x20", {})[a] = v5 / v20.replace(0, np.nan)
        # ---- Batch C: trend / oscillator ----
        up = r.clip(lower=0)
        down = (-r).clip(lower=0)
        ru = up.rolling(5).mean()
        rd = down.rolling(5).mean()
        F.setdefault("rsi_5", {})[a] = 100 * ru / (ru + rd).replace(0, np.nan)
        mom20 = c / c.shift(20) - 1.0
        F.setdefault("trend_strength_20", {})[a] = mom20.abs() / r.abs().rolling(20).sum().replace(0, np.nan)
        F.setdefault("mom_5d_skip2", {})[a] = c.shift(2) / c.shift(7) - 1.0
        lo10 = c.rolling(10).min()
        hi10 = c.rolling(10).max()
        F.setdefault("hl_pos_10", {})[a] = (c - lo10) / (hi10 - lo10).replace(0, np.nan)
    # ---- Batch D: volume factors ----
    for a, v in vols.items():
        if v is None:
            continue
        va20 = v.rolling(20).mean()
        va60 = v.rolling(60).mean()
        F.setdefault("volume_trend_20x60", {})[a] = va20 / va60.replace(0, np.nan)
        rc = closes[a].pct_change()
        F.setdefault("pv_corr_20", {})[a] = rc.rolling(20).corr(v)

    for k in F:
        F[k] = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in F[k].items()}
    return F


F = build_all()

# ---- active library replication (usdcny_beta_60) ----
usdcny_r = OBS["USDCNY"].reindex(fclose.index).ffill().pct_change()
LIB = {}
for a, c in closes.items():
    r = c.pct_change()
    LIB.setdefault("usdcny_beta_60", {})[a] = r.rolling(60).cov(usdcny_r) / usdcny_r.rolling(60).var()
lib_df = pd.DataFrame(LIB["usdcny_beta_60"]).stack()
lib_df = lib_df[lib_df.notna()]


def lib_corr(factor):
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    both = fdf.index.intersection(lib_df.index)
    if len(both) < 200:
        return np.nan, 0
    rho = pearsonr(fdf.loc[both].values, lib_df.loc[both].values)[0]
    fpanel = pd.DataFrame(factor)
    lpanel = pd.DataFrame(LIB["usdcny_beta_60"])
    dmax, nd = 0.0, 0
    for dt in fpanel.index.intersection(lpanel.index):
        x = fpanel.loc[dt].dropna()
        y = lpanel.loc[dt].reindex(x.index).dropna()
        b = x.index.intersection(y.index)
        if len(b) >= MIN_ASSETS:
            rr = spearmanr(x[b], y[b])[0]
            dmax = max(dmax, abs(rr))
            nd += 1
    return (float(rho) if np.isfinite(rho) else np.nan), dmax


print("\n=== CANDIDATE VALIDATION (primary horizon 10d) ===")
results = {}
for name, fac in sorted(F.items()):
    fdf = pd.DataFrame(fac)
    n_valid_assets = int((fdf.notna().sum() > 200).sum())
    cov = float(fdf.notna().sum().sum() / (fdf.shape[0] * fdf.shape[1]))
    ic10 = metrics(ic_series(fdf, fwd10))
    if not np.isfinite(ic10["ic"]) or ic10["n"] < MIN_IC_DATES:
        print(f"[{name:24s}] insufficient IC dates n={ic10['n']}; skip")
        continue
    ic1 = metrics(ic_series(fdf, fwd1))
    ic3 = metrics(ic_series(fdf, fwd3))
    ic5 = metrics(ic_series(fdf, fwd5))
    ic20 = metrics(ic_series(fdf, fwd20))
    rho_pearson, dmax = lib_corr(fac)
    gate_ic = abs(ic10["ic"]) >= IC_THR
    gate_icir = abs(ic10["icir"]) >= ICIR_THR
    gate_rho = np.isfinite(rho_pearson) and abs(rho_pearson) < RHO_THR
    ok = gate_ic and gate_icir and gate_rho
    # recent 12m and 6m checks
    recent = fdf.loc[fdf.index >= "2026-06-01"]
    ic_recent = metrics(ic_series(recent, fwd10.reindex(recent.index))) if len(recent) > 30 else None
    recent6 = fdf.loc[fdf.index >= "2026-12-01"]
    ic_recent6 = metrics(ic_series(recent6, fwd10.reindex(recent6.index))) if len(recent6) > 30 else None
    results[name] = dict(ic10=ic10, ic1=ic1, ic3=ic3, ic5=ic5, ic20=ic20,
                         rho_pearson=rho_pearson, dmax=dmax, cov=cov,
                         n_valid_assets=n_valid_assets, gate=ok,
                         recent=ic_recent, recent6=ic_recent6)
    print(f"[{name:24s}] IC10={ic10['ic']:+.4f} ICIR10={ic10['icir']:+.4f} n={ic10['n']:4d} hit={ic10['hit']:.3f} | "
          f"IC1={ic1['ic']:+.4f} IC3={ic3['ic']:+.4f} IC5={ic5['ic']:+.4f} IC20={ic20['ic']:+.4f} | "
          f"cov={cov:.3f} nA={n_valid_assets:2d} | rho_lib={rho_pearson:+.3f} dmax={dmax:.3f} | "
          f"{'PASS' if ok else 'fail'}"
          + (f" | R12m IC={ic_recent['ic']:+.4f}/{ic_recent['n']} R6m IC={ic_recent6['ic']:+.4f}/{ic_recent6['n']}"
             if ic_recent and ic_recent6 else ""))

# ---- revalidation of active library factor ----
print("\n=== ACTIVE LIBRARY REVALIDATION (usdcny_beta_60) ===")
libdf = pd.DataFrame(LIB["usdcny_beta_60"])
for lbl, sub in [("full", libdf), ("12m", libdf.loc[libdf.index >= "2026-06-01"]),
                 ("6m", libdf.loc[libdf.index >= "2026-12-01"])]:
    m = metrics(ic_series(sub, fwd10.reindex(sub.index)))
    print(f"usdcny_beta_60 [{lbl:4s}] IC10={m['ic']:+.4f} ICIR10={m['icir']:+.4f} n={m['n']} hit={m['hit']:.3f}")

print("\nDONE")
