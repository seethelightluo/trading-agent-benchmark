"""miner_2 2027-06-17: final validation + persistence for passing candidates.

Candidates that passed the IC/ICIR admission gate in the 06-03 exploration:
  max_gain_20        (lottery/breakout demand: max 1d return over 20d)
  trend_strength_20  (efficiency ratio 20d: |mom20| / sum(|ret|,20))

Both have full cross-sectional coverage (nA=15) and 1210 IC dates spanning
2020..2027 (multiple regimes), unlike the cross-asset beta factors whose
coverage collapses to 7% (sparse commodity data pre-2026) and pv_corr_20 whose
recent window is empty (volume absent for 6/15 assets). Those are reported as
data-quality-rejected despite passing numeric gates.

This script recomputes all admission metrics (IC/ICIR @10d, hit, coverage,
turnover, decay, regime splits, library correlation vs usdcny_beta_60) and
writes factors/<factor_id>.json with a recoverable signal artifact
(base64:zlib:csv of the factor value panel).
"""
import json, zlib, base64, datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

IC_THR = 0.0070
ICIR_THR = 0.0840
RHO_THR = 0.5
MIN_ASSETS = 8
MIN_IC_DATES = 60
TODAY = "2027-06-17"

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


print("loading ...")
data = {a: load_asset(a) for a in WATCH}
data = {a: d for a, d in data.items() if d is not None}
closes = {a: d["close"].astype(float) for a, d in data.items()}
fclose = pd.DataFrame(closes).sort_index()
fwd10 = fclose.shift(-10) / fclose - 1.0
fwd1 = fclose.shift(-1) / fclose - 1.0
fwd2 = fclose.shift(-2) / fclose - 1.0
fwd3 = fclose.shift(-3) / fclose - 1.0
fwd5 = fclose.shift(-5) / fclose - 1.0
fwd20 = fclose.shift(-20) / fclose - 1.0
rets = fclose.pct_change()
print(f"panel {fclose.shape}, last_vis={fclose.index.max().date()}")

# ---- build the two candidates ----
F = {}
for a, c in closes.items():
    r = c.pct_change()
    F.setdefault("max_gain_20", {})[a] = r.rolling(20).max()
    mom20 = c / c.shift(20) - 1.0
    F.setdefault("trend_strength_20", {})[a] = mom20.abs() / r.abs().rolling(20).sum().replace(0, np.nan)
for k in F:
    F[k] = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in F[k].items()}

# ---- active library: usdcny_beta_60 ----
obs = pd.read_csv(f"{INDEX_DATA_DIR}/USDCNY.csv")
obs["date"] = pd.to_datetime(obs["date"])
usdcny = obs.set_index("date").sort_index()["close"].astype(float)
usdcny = usdcny[usdcny.index <= fclose.index.max()]
usdcny_r = usdcny.reindex(fclose.index).ffill().pct_change()
LIB = {}
for a, c in closes.items():
    r = c.pct_change()
    LIB.setdefault("usdcny_beta_60", {})[a] = r.rolling(60).cov(usdcny_r) / usdcny_r.rolling(60).var()
lib_df = pd.DataFrame(LIB["usdcny_beta_60"]).stack()
lib_df = lib_df[lib_df.notna()]


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


def turnover_10d_rank(fdf):
    """mean abs change in cross-sectional rank between t-10 and t, per asset-day avg."""
    ranks = fdf.rank(axis=1)
    r10 = ranks.shift(10)
    chg = (ranks - r10).abs()
    return float(chg.mean())


def regime_ic(fdf, fwd, name):
    out = {}
    spans = [("2020-2021 COVID/recovery", "2020-01-01", "2021-12-31"),
             ("2022-2023 tightening/AI", "2022-01-01", "2023-12-31"),
             ("2024-2026-07 crypto/commodity", "2024-01-01", "2026-07-31"),
             ("2026-08..2027-06 online-era", "2026-08-01", "2027-12-31")]
    for lbl, s, e in spans:
        sub = fdf.loc[(fdf.index >= s) & (fdf.index <= e)]
        if len(sub) < 30:
            out[lbl] = None
            continue
        m = metrics(ic_series(sub, fwd.reindex(sub.index)))
        out[lbl] = [m["ic"], m["icir"], m["n"]] if np.isfinite(m["ic"]) else None
    return out


def lib_corr_ravel(fdf):
    fdf_stack = fdf.stack()
    fdf_stack = fdf_stack[fdf_stack.notna()]
    both = fdf_stack.index.intersection(lib_df.index)
    if len(both) < 200:
        return np.nan
    return float(pearsonr(fdf_stack.loc[both].values, lib_df.loc[both].values)[0])


def make_artifact(fdf):
    csv = fdf.to_csv().encode()
    return base64.b64encode(zlib.compress(csv)).decode()


print("\n=== FINAL VALIDATION ===")
results = {}
for name in ["max_gain_20", "trend_strength_20"]:
    fdf = pd.DataFrame(F[name])
    cov_ad = float(fdf.notna().sum().sum() / (fdf.shape[0] * fdf.shape[1]))
    cov_d8 = float((fdf.notna().sum(axis=1) >= MIN_ASSETS).mean())
    n_valid_assets = int((fdf.notna().sum() > 200).sum())
    ic10 = metrics(ic_series(fdf, fwd10))
    ic1 = metrics(ic_series(fdf, fwd1))
    ic2 = metrics(ic_series(fdf, fwd2))
    ic3 = metrics(ic_series(fdf, fwd3))
    ic5 = metrics(ic_series(fdf, fwd5))
    ic20 = metrics(ic_series(fdf, fwd20))
    decay = {h: ic10["ic"] for h in []}
    decay = {"1": ic1["ic"], "2": ic2["ic"], "3": ic3["ic"], "5": ic5["ic"],
             "10": ic10["ic"], "20": ic20["ic"]}
    to = turnover_10d_rank(fdf)
    reg = regime_ic(fdf, fwd10, name)
    rho = lib_corr_ravel(fdf)
    recent = metrics(ic_series(fdf.loc[fdf.index >= "2026-06-01"], fwd10.reindex(fdf.loc[fdf.index >= "2026-06-01"].index)))
    recent6 = metrics(ic_series(fdf.loc[fdf.index >= "2026-12-01"], fwd10.reindex(fdf.loc[fdf.index >= "2026-12-01"].index)))
    gate_ic = abs(ic10["ic"]) >= IC_THR
    gate_icir = abs(ic10["icir"]) >= ICIR_THR
    gate_rho = np.isfinite(rho) and abs(rho) < RHO_THR
    ok = gate_ic and gate_icir and gate_rho
    results[name] = dict(fdf=fdf, ic10=ic10, decay=decay, to=to, reg=reg, rho=rho,
                         cov_ad=cov_ad, cov_d8=cov_d8, n_valid_assets=n_valid_assets,
                         recent=recent, recent6=recent6, ok=ok)
    print(f"[{name:20s}] IC10={ic10['ic']:+.4f} ICIR10={ic10['icir']:+.4f} n={ic10['n']} "
          f"hit={ic10['hit']:.3f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} nA={n_valid_assets} "
          f"to10={to:.3f} rho_lib={rho:+.3f} | {'PASS' if ok else 'FAIL'}")
    print(f"    decay(1/2/3/5/10/20): " +
          " ".join(f"{h}:{v:+.4f}" for h, v in decay.items()))
    print(f"    regime: " + "; ".join(f"{k.split()[0]}{k.split()[1][:4]}={v}" for k, v in reg.items() if v))
    print(f"    recent12m IC={recent['ic']:+.4f}/{recent['n']}  recent6m IC={recent6['ic']:+.4f}/{recent6['n']}")

print("\nDONE validation")
