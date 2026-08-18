"""miner_3 2026-12-03: explore fresh candidate factors on the 15-asset cross-asset
universe. Data visible through 2026-12-02 (previous completed trading day).

Candidates (per-asset cross-sectional signals):
  1) usdjpy_beta_cond_60x20   : -beta(ret,USDJPY,60)*(USDJPY 20d move)
  2) eurusd_beta_cond_60x20   : -beta(ret,EURUSD,60)*(EURUSD 20d move)
  3) usdcny_cond_60x20        : -beta(ret,USDCNY,60)*(USDCNY 20d move) (conditional vs plain usdcny_beta_60)
  4) cn10y_beta_cond_60x20    : -beta(ret,CN10Y,60)*(CN10Y 20d move)
  5) us10y_dln_beta_60x20     : beta(ret, d(US10Y level),60)*(d(US10Y) 20d)
  6) crisis_beta_60           : SPX-down-day corr - SPX-up-day corr (60d)
  7) vol_trend_60x10          : (vol10 - vol60)/vol60
  8) mom_vol_ratio_10x20      : 10d mom / 20d vol (risk-adjusted momentum)
  9) hl_range_60              : 60d (max-min)/close
 10) maxdd_30                 : rolling 30d max drawdown (negative)
 11) skew_20                  : 20d return skewness
 12) comm_ratio_beta_60       : beta(ret, XAU/WTI ratio ret, 60)
 13) mom_120d_skip5           : 120d momentum skip5 (re-check vs current library)
 14) rel_mom_20               : 20d momentum - cross-sectional mean

Admission gates: |IC10|>=0.0070 and |ICIR10|>=0.0840; |rho| vs active library (usdcny_beta_60) < 0.5.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

IC_THR = 0.0070
ICIR_THR = 0.0840
RHO_THR = 0.5
MIN_ASSETS = 8
MIN_IC_DATES = 60

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
INDEX_DATA_DIR = "../persistent/index_data/"


def load_asset(symbol, days=3200):
    from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None:
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


def load_obs(name):
    df = pd.read_csv(f"{INDEX_DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").sort_index()["close"].astype(float)
    return s[s.index <= last_vis]


print("loading data ...")
data = {a: load_asset(a, days=3200) for a in WATCH}
data = {a: d for a, d in data.items() if d is not None}
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
last_vis = max(d.index.max() for d in data.values())
print(f"loaded {len(data)}/15 instruments; last_vis={last_vis.date()}")

OBS = {m: load_obs(m) for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
print(f"macro last dates: { {m: s.index.max().date() for m, s in OBS.items()} }")


def ic_series(factor, fwd, min_assets=MIN_ASSETS):
    """Cross-sectional Spearman IC per date (both indexed by date)."""
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
        return dict(ic=np.nan, icir=np.nan, n=n, hit=np.nan, t=np.nan, dates_ge8=np.nan)
    ic = float(ics.mean())
    sd = float(ics.std(ddof=1)) if n > 1 else 0.0
    icir = ic / sd if sd > 0 else np.nan
    hit = float((ics > 0).mean())
    t = ic / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return dict(ic=ic, icir=icir, n=n, hit=hit, t=t)


# forward returns panel
fclose = pd.DataFrame(closes).sort_index()
fwd10 = fclose.shift(-10) / fclose - 1.0
fwd1 = fclose.shift(-1) / fclose - 1.0
fwd3 = fclose.shift(-3) / fclose - 1.0
fwd20 = fclose.shift(-20) / fclose - 1.0

rets = fclose.pct_change()
rspx = rets["SPX"]


def cond_beta(c, macro_r, win=60, dwin=20):
    beta = c.pct_change().rolling(win).cov(macro_r) / macro_r.rolling(win).var()
    move = macro_r.rolling(dwin).mean() * dwin  # approx cumulative move
    move2 = macro_r.rolling(dwin).sum()
    return -beta * move2


def build_all():
    F = {}
    usdjpy = OBS["USDJPY"].reindex(fclose.index).ffill().pct_change()
    eurusd = OBS["EURUSD"].reindex(fclose.index).ffill().pct_change()
    usdcny = OBS["USDCNY"].reindex(fclose.index).ffill().pct_change()
    vix = OBS["VIX"].reindex(fclose.index).ffill()
    cn10 = closes["CN10Y"]
    us10 = closes["US10Y"]
    cn10r = cn10.pct_change()
    us10r = us10.pct_change()
    us10d = us10.diff()
    xauwti = (closes["XAU"] / closes["WTI"])

    for a, c in closes.items():
        r = c.pct_change()
        F.setdefault("usdjpy_beta_cond_60x20", {})[a] = cond_beta(c, usdjpy)
        F.setdefault("eurusd_beta_cond_60x20", {})[a] = cond_beta(c, eurusd)
        F.setdefault("usdcny_cond_60x20", {})[a] = cond_beta(c, usdcny)
        F.setdefault("cn10y_beta_cond_60x20", {})[a] = cond_beta(c, cn10r)
        F.setdefault("us10y_dln_beta_60x20", {})[a] = -c.pct_change().rolling(60).cov(us10d) / us10d.rolling(60).var() * us10d.rolling(20).sum()
        # crisis beta: SPX-down-day correlation minus up-day correlation (60d)
        down = rspx < 0
        up = rspx > 0
        rc = r.rolling(60).corr(rspx)
        cd = r[down].rolling(60).corr(rspx[down])
        cu = r[up].rolling(60).corr(rspx[up])
        F.setdefault("crisis_beta_60", {})[a] = cd - cu
        v20 = r.rolling(20).std()
        v60 = r.rolling(60).std()
        F.setdefault("vol_trend_60x10", {})[a] = (v20 - v60) / v60.replace(0, np.nan)
        mom10 = c.shift(5) / c.shift(15) - 1.0
        F.setdefault("mom_vol_ratio_10x20", {})[a] = mom10 / v20.replace(0, np.nan)
        F.setdefault("hl_range_60", {})[a] = (c.rolling(60).max() - c.rolling(60).min()) / c
        rollmax = c.rolling(30).max()
        F.setdefault("maxdd_30", {})[a] = c / rollmax - 1.0
        F.setdefault("skew_20", {})[a] = r.rolling(20).skew()
        xr = xauwti.pct_change()
        F.setdefault("comm_ratio_beta_60", {})[a] = r.rolling(60).cov(xr) / xr.rolling(60).var()
        F.setdefault("mom_120d_skip5", {})[a] = c.shift(5) / c.shift(125) - 1.0
        F.setdefault("rel_mom_20", {})[a] = c / c.shift(20) - 1.0
    # rel_mom_20: cross-sectional demean
    tmp = pd.DataFrame(F["rel_mom_20"])
    F["rel_mom_20"] = {a: (tmp[a] - tmp.mean(axis=1)) for a in tmp.columns}
    for k in F:
        F[k] = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in F[k].items()}
    return F


F = build_all()

# ---- active library replication (usdcny_beta_60 exactly as persisted) ----
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
        return np.nan, {}, 0
    rho = pearsonr(fdf.loc[both].values, lib_df.loc[both].values)[0]
    # datewise max |spearman|
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
    return (float(rho) if np.isfinite(rho) else np.nan), {}, nd


print("\n=== CANDIDATE VALIDATION (primary horizon 10d) ===")
results = {}
for name, fac in sorted(F.items()):
    fdf = pd.DataFrame(fac)
    # coverage
    n_valid_assets = int((fdf.notna().sum() > 200).sum())
    cov = float(fdf.notna().sum().sum() / (fdf.shape[0] * fdf.shape[1]))
    # IC
    ic10 = metrics(ic_series(fdf, fwd10))
    if not np.isfinite(ic10["ic"]) or ic10["n"] < MIN_IC_DATES:
        print(f"[{name}] insufficient IC dates n={ic10['n']}; skip")
        continue
    ic1 = metrics(ic_series(fdf, fwd1))
    ic3 = metrics(ic_series(fdf, fwd3))
    ic20 = metrics(ic_series(fdf, fwd20))
    rho_pearson, _, n_dates_lib = lib_corr(fac)
    # datewise max spearman vs library
    fpanel = pd.DataFrame(fac)
    lpanel = pd.DataFrame(LIB["usdcny_beta_60"])
    dmax = 0.0
    nd = 0
    for dt in fpanel.index.intersection(lpanel.index):
        x = fpanel.loc[dt].dropna()
        y = lpanel.loc[dt].reindex(x.index).dropna()
        b = x.index.intersection(y.index)
        if len(b) >= MIN_ASSETS:
            rr = spearmanr(x[b], y[b])[0]
            dmax = max(dmax, abs(rr))
            nd += 1
    gate_ic = abs(ic10["ic"]) >= IC_THR
    gate_icir = abs(ic10["icir"]) >= ICIR_THR
    gate_rho = np.isfinite(rho_pearson) and abs(rho_pearson) < RHO_THR
    ok = gate_ic and gate_icir and gate_rho
    # recent 12m check
    recent = fdf.loc[fdf.index >= "2025-12-01"] if fdf.index.max() >= pd.Timestamp("2025-12-01") else fdf.iloc[0:0]
    ic_recent = metrics(ic_series(recent, fwd10.reindex(recent.index))) if len(recent) > 30 else None
    results[name] = dict(ic10=ic10, ic1=ic1, ic3=ic3, ic20=ic20,
                         rho_pearson=rho_pearson, dmax_spearman=dmax, n_lib_dates=nd,
                         cov=cov, n_valid_assets=n_valid_assets, gate=ok,
                         recent=(ic_recent if ic_recent else {}), n_recent=(len(ic_recent) if ic_recent else 0))
    print(f"[{name:26s}] IC10={ic10['ic']:+.4f} ICIR10={ic10['icir']:+.4f} n={ic10['n']:4d} "
          f"hit={ic10['hit']:.3f} | IC1={ic1['ic']:+.4f} IC3={ic3['ic']:+.4f} IC20={ic20['ic']:+.4f} | "
          f"cov={cov:.3f} nA={n_valid_assets:2d} | rho_lib={rho_pearson:+.3f} dmax={dmax:.3f} | "
          f"{'PASS' if ok else 'fail'}"
          + (f" | R12m IC={ic_recent['ic']:+.4f} n={len(ic_recent)}" if ic_recent else ""))

print("\n=== REGIME SPLITS (h=10) for PASS candidates ===")
regimes = {"2020-2021": ("2020-01-01", "2021-12-31"),
           "2022-2023": ("2022-01-01", "2023-12-31"),
           "2024-2025": ("2024-01-01", "2025-12-31"),
           "2026": ("2026-01-01", None)}
for name, r in results.items():
    if not r["gate"]:
        continue
    fdf = pd.DataFrame(F[name])
    parts = []
    for rn, (s, e) in regimes.items():
        sub = fdf.loc[fdf.index >= pd.Timestamp(s)]
        if e:
            sub = sub.loc[sub.index <= pd.Timestamp(e)]
        m = metrics(ic_series(sub, fwd10.reindex(sub.index)))
        parts.append(f"{rn}:IC={m['ic']:+.4f}/ICIR={m['icir']:+.3f}/n={m['n']}")
    print(f"[{name:26s}] " + " | ".join(parts))

print("\nDONE")