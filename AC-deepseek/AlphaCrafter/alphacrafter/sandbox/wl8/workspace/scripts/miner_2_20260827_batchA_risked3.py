"""miner_2 2026-08-27 -- Batch A: risk-adjusted reversal & orthogonal dynamics.
Motivation: current regime (per 2026-07-30 ensemble notes) is mixed/corrective with
HIGH cross-sectional dispersion and LOW cross-asset correlation. The active library
holds only usdcny_beta_60 (sparse, 14% coverage). We seek factors that:
  (a) pass |IC|>=0.007 & |ICIR|>=0.084 on the 15-asset cross-asset universe
  (b) are orthogonal to usdcny_beta_60 (pooled |rho|<0.5) so the gate keeps them
  (c) are regime-robust across 6m/3m recent windows.

Candidates (one idea per function):
  1. rv_ret_5   : 5d return / 20d realized vol  (risk-adjusted short-term reversal)
  2. rv_ret_10  : 10d return / 20d realized vol (risk-adjusted reversal, slower)
  3. rv_ret_20  : 20d return / 20d realized vol
  4. mom_resid_10 : 10d residual return after 60d beta regression on SPX
                   (idiosyncratic momentum; expected orthogonal to raw mom)
  5. yld_beta_dense_60 : 60d beta of asset returns to US10Y changes (dense variant
                   - library holds only the CONDITIONAL sparse version)
  6. worst5_bounce : -1 * (min 5d return)/20d vol  (deep-pain rebound)
Data: CSVs under ../persistent, truncated at 2026-08-26 (visible through previous
completed trading day; no lookahead). Validation horizon 10d primary.
"""
import json
import base64
import zlib
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
END = pd.Timestamp("2026-08-26")
START = pd.Timestamp("2021-01-01")  # skip warmup early noisy period for summary
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS = 8


def load_panel(suffix=""):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[(df["date"] <= END)].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float)
        highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    return (pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens),
            pd.DataFrame(highs), pd.DataFrame(lows))


def load_macro():
    out = {}
    for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
        df = pd.read_csv(f"{INDEX_DIR}/{k}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        out[k] = df["close"].astype(float)
    return out


close, vol, open_, high, low = load_panel()
macro = load_macro()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()} "
      f"{len(close)} dates x {close.shape[1]} assets")


def dense_per_asset():
    d = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        d[a] = {"close": close[a].reindex(idx), "vol": vol[a].reindex(idx),
                "open": open_[a].reindex(idx), "high": high[a].reindex(idx),
                "low": low[a].reindex(idx)}
    return d


DENSE = dense_per_asset()


def factor_panel(fn, **params):
    out = {}
    for a in ASSETS:
        dc = DENSE[a]
        try:
            s = fn(dc["close"], dc["vol"], dc["open"], dc["high"], dc["low"],
                   macro, **params)
            out[a] = pd.Series(np.asarray(s, dtype=float), index=dc["close"].index)
            out[a] = out[a].reindex(close.index)
        except Exception as e:
            print(f"  [err] {a}: {e}")
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)


def fwd_returns(horizon):
    out = {}
    for a in ASSETS:
        c = DENSE[a]["close"]
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)


def ic_series(factor, fwd):
    dates, ics = [], []
    for dt in factor.index:
        x = factor.loc[dt]
        y = fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            r = spearmanr(x[m], y[m])
            if np.isfinite(r.statistic):
                ics.append(r.statistic)
                dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def metrics(factor, h=10, label=""):
    fr = fwd_returns(h)
    ic = ic_series(factor, fr)
    stats = {}
    for name, mask in [("full", ic.index >= START), ("6m", ic.index >= END - pd.Timedelta(days=183)),
                       ("3m", ic.index >= END - pd.Timedelta(days=92))]:
        s = ic[mask]
        if len(s) >= 3:
            m_ic = float(s.mean())
            stats[name] = (m_ic, float(s.mean() / s.std()) if s.std() > 0 else np.nan,
                           int(len(s)), float((s > 0).mean()))
        else:
            stats[name] = (np.nan, np.nan, int(len(s)), np.nan)
    to = float(factor.rank(axis=1).diff(10).abs().mean(axis=1).mean())
    cov_ad = float(factor.notna().sum().sum() / (factor.shape[0] * factor.shape[1]))
    cov_ge8 = float((factor.notna().sum(axis=1) >= MIN_ASSETS).mean())
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        s = ic_series(factor, fwd_returns(hh))
        s = s[s.index >= START]
        decay[hh] = float(s.mean()) if len(s) else np.nan
    return {"stats": stats, "turnover_10d": to, "cov_ad": cov_ad,
            "cov_ge8": cov_ge8, "decay": decay}


def gate_pass(stats):
    ic_full, icir_full = stats["full"][0], stats["full"][1]
    return abs(ic_full) >= IC_GATE and abs(icir_full) >= ICIR_GATE and np.isfinite(icir_full)


# ---------------- factor definitions ----------------
def _beta(asset_ret, driver_ret, win):
    cov = asset_ret.rolling(win).cov(driver_ret)
    var = driver_ret.rolling(win).var()
    return cov / var


def f_rv_ret(c, v, o, h, l, m, w_ret=5, w_vol=20):
    ret = c.pct_change(w_ret)
    v20 = c.pct_change().rolling(w_vol).std()
    return ret / v20


def f_rv_ret10(c, v, o, h, l, m, w_ret=10, w_vol=20):
    return f_rv_ret(c, v, o, h, l, m, w_ret, w_vol)


def f_rv_ret20(c, v, o, h, l, m, w_ret=20, w_vol=20):
    return f_rv_ret(c, v, o, h, l, m, w_ret, w_vol)


def f_mom_resid_10(c, v, o, h, l, m, win=60, hz=10):
    """10d idiosyncratic (market-residual) momentum: residual of 10d return over
    60d beta regression on SPX. Uses only the asset's own dense calendar."""
    spx = close["SPX"].reindex(c.index)
    r_a = c.pct_change()
    r_s = spx.pct_change()
    beta = r_a.rolling(win).cov(r_s) / r_s.rolling(win).var()
    resid_ret = r_a - beta * r_s
    return resid_ret.rolling(hz).sum()


def f_yld_beta_dense_60(c, v, o, h, l, m, win=60):
    us10 = close["US10Y"].reindex(c.index)
    d_y = us10.diff()
    return _beta(c.pct_change(), d_y, win)


def f_worst5_bounce(c, v, o, h, l, m, w=5, w_vol=20):
    """-1 * min 5d return / 20d vol: deeper pain -> higher value."""
    r = c.pct_change()
    wret = r.rolling(w).sum()
    v20 = r.rolling(w_vol).std()
    return -wret / v20


# ---------------- library correlation ----------------
def load_lib_panels():
    lib = {}
    for fid in ["usdcny_beta_60"]:
        try:
            d = json.load(open(f"factors/{fid}.json"))
            if d.get("validation", {}).get("status") != "EFFECTIVE":
                continue
            raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
            p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                            index_col=0, parse_dates=True)
            p.index = pd.DatetimeIndex(p.index)
            lib[fid] = p
        except Exception as e:
            print(f"  [warn] lib load {fid}: {e}")
    return lib


LIB = load_lib_panels()
print(f"Active library panels: {list(LIB.keys())}")


def max_lib_corr(panel):
    best = {}
    for fid, lp in LIB.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        best[fid] = round(rho, 4)
    return best


# ---------------- run ----------------
CANDIDATES = [
    ("rv_ret_5", f_rv_ret, {"w_ret": 5, "w_vol": 20}),
    ("rv_ret_10", f_rv_ret10, {"w_ret": 10, "w_vol": 20}),
    ("rv_ret_20", f_rv_ret20, {"w_ret": 20, "w_vol": 20}),
    ("mom_resid_10", f_mom_resid_10, {}),
    ("yld_beta_dense_60", f_yld_beta_dense_60, {}),
    ("worst5_bounce", f_worst5_bounce, {}),
]

results = {}
for name, fn, p in CANDIDATES:
    panel = factor_panel(fn, **p)
    res = metrics(panel, label=name)
    rho = max_lib_corr(panel)
    res["rho_vs_lib"] = rho
    res["max_abs_library_correlation"] = max([abs(x) for x in rho.values()], default=0.0)
    results[name] = res
    ic_f, icir_f, n_f, hit_f = res["stats"]["full"]
    passed = gate_pass(res["stats"])
    print(f"\n=== {name} ===")
    print(f"  full: IC={ic_f:.4f} ICIR={icir_f:.4f} n={n_f} hit={hit_f:.3f} {'PASS' if passed else 'FAIL'}")
    for k in ["3m", "6m"]:
        ic_, icir_, n_, hit_ = res["stats"][k]
        print(f"  {k}: IC={ic_:.4f} ICIR={icir_:.4f} n={n_} hit={hit_:.3f}")
    print(f"  turnover_10d={res['turnover_10d']:.3f} cov_ad={res['cov_ad']:.3f} "
          f"cov_ge8={res['cov_ge8']:.3f}")
    print(f"  decay: {res['decay']}")
    print(f"  rho_vs_lib={rho}  max_abs={res['max_abs_library_correlation']:.4f}")

print("\n=== SUMMARY ===")
for name, res in results.items():
    ic_f, icir_f, n_f, hit_f = res["stats"]["full"]
    passed = gate_pass(res["stats"])
    print(f"{name}: PASS={passed} IC={ic_f:.4f} ICIR={icir_f:.4f} n={n_f} "
          f"max_rho_lib={res['max_abs_library_correlation']:.3f}")