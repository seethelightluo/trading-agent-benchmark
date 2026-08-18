"""miner_2 2026-08-27 -- rv_ret family tuning: window sweep + sibling/lib correlation.
Decide which single risk-adjusted-return factor to persist (they are same-family
so persisting two would trigger pairwise-conflict eviction). Evaluated on the same
panel/validation window as batch A (data through 2026-08-26, primary horizon 10d).
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
START = pd.Timestamp("2021-01-01")
IC_GATE, ICIR_GATE, MIN_ASSETS = 0.0070, 0.0840, 8


def load_panel():
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        closes[a] = df["close"].astype(float)
    return pd.DataFrame(closes)


close = load_panel()
DENSE = {a: close[a].dropna() for a in ASSETS}


def factor_panel(w_ret, w_vol):
    out = {}
    for a in ASSETS:
        c = DENSE[a]
        r = c.pct_change()
        f = (c.pct_change(w_ret) / r.rolling(w_vol).std()).reindex(close.index)
        out[a] = f
    return pd.DataFrame(out)


def fwd_returns(horizon):
    out = {}
    for a in ASSETS:
        c = DENSE[a]
        out[a] = (c.shift(-horizon) / c - 1.0).reindex(close.index)
    return pd.DataFrame(out)


def ic_series(factor, fwd):
    dates, ics = [], []
    for dt in factor.index:
        x, y = factor.loc[dt], fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            r = spearmanr(x[m], y[m])
            if np.isfinite(r.statistic):
                ics.append(r.statistic)
                dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def load_lib_panels():
    lib = {}
    for fid in ["usdcny_beta_60"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                        index_col=0, parse_dates=True)
        p.index = pd.DatetimeIndex(p.index)
        lib[fid] = p
    return lib


LIB = load_lib_panels()


def pooled_rho(p1, p2):
    common = p1.index.intersection(p2.index)
    cols = [c for c in p1.columns if c in p2.columns]
    a = p1.loc[common, cols].values.ravel()
    b = p2.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 200 else np.nan


def summarize(name, panel):
    ic = ic_series(panel, fwd_returns(10))
    ic = ic[ic.index >= START]
    rows = []
    for lab, m in [("full", ic.index >= START),
                   ("1y", ic.index >= END - pd.Timedelta(days=366)),
                   ("6m", ic.index >= END - pd.Timedelta(days=183)),
                   ("3m", ic.index >= END - pd.Timedelta(days=92))]:
        s = ic[m]
        if len(s) >= 3:
            rows.append((lab, float(s.mean()), float(s.mean() / s.std()), int(len(s)),
                         float((s > 0).mean())))
        else:
            rows.append((lab, np.nan, np.nan, int(len(s)), np.nan))
    rho_lib = pooled_rho(panel, LIB["usdcny_beta_60"])
    to = float(panel.rank(axis=1).diff(10).abs().mean(axis=1).mean())
    print(f"{name}: " + " | ".join(
        f"{lab} IC={ic_:.4f} ICIR={icir_:.4f} n={n} hit={hit:.3f}" for lab, ic_, icir_, n, hit in rows)
          + f" | rho_lib={rho_lib:.3f} TO={to:.2f}")


panels = {}
for wr in [5, 10, 15, 20, 30]:
    for wv in [20, 40]:
        name = f"rv_{wr}_{wv}"
        p = factor_panel(wr, wv)
        panels[name] = p
        summarize(name, p)

print("\nSibling pooled correlations (rv_10_20 vs others):")
p10 = panels["rv_10_20"]
for name, p in panels.items():
    print(f"  rv_10_20 vs {name}: {pooled_rho(p10, p):.3f}")