"""Shared validation harness for miner_1 research cycle (validation through 2026-09-09).

Loads the 15-asset cross-asset tradable universe from ../persistent CSVs (truncated at
END=2026-09-09, i.e., data visible through the previous completed trading day - no lookahead)
plus observation-only macro series. Computes cross-sectional rank IC / ICIR / coverage /
turnover / decay / regime splits / library correlation (pooled Spearman vs the live
EFFECTIVE library, currently usdcny_beta_60).

Admission gates (15-instrument universe):
  |IC|  >= 0.0070
  |ICIR| >= 0.0840
  max_abs_library_correlation < 0.5 (pooled Spearman, gate recomputes from artifacts)
"""
import json
import base64
import zlib
import io
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
END = pd.Timestamp("2026-09-09")
START = pd.Timestamp("2020-01-02")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
RHO_GATE = 0.5
MIN_ASSETS = 8


def load_panel():
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
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
        df = df[~df.index.duplicated(keep="last")]
        out[k] = df["close"].astype(float)
    return out


def dense_per_asset(close, vol, open_, high, low):
    d = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        d[a] = {"close": close[a].reindex(idx), "vol": vol[a].reindex(idx),
                "open": open_[a].reindex(idx), "high": high[a].reindex(idx),
                "low": low[a].reindex(idx)}
    return d


def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    """Apply factor fn per asset on its dense calendar, reindex to union panel."""
    dense = dense_per_asset(close, vol, open_, high, low)
    out = {}
    for a in ASSETS:
        dc = dense[a]
        try:
            s = fn(dc["close"], dc["vol"], dc["open"], dc["high"], dc["low"], macro, **params)
            out[a] = pd.Series(np.asarray(s).ravel(), index=dc["close"].index).reindex(close.index)
        except Exception as e:
            print(f"  [warn] {a} factor fail: {e}")
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)


def validate_factor(panel, close, horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10,
                    min_assets=MIN_ASSETS, start=START, end=END):
    fdf = panel
    ics_main = []
    decay = {}
    for h in horizons:
        fwd = close.pct_change(h).shift(-h)
        ics = []
        dates = fdf.index[(fdf.index >= start) & (fdf.index <= end)]
        for dt in dates:
            fv = fdf.loc[dt]
            rv = fwd.loc[dt]
            m = fv.notna() & rv.notna()
            if m.sum() < min_assets:
                continue
            ic, _ = spearmanr(fv[m], rv[m])
            if np.isfinite(ic):
                ics.append((dt, ic))
        if not ics:
            decay[h] = np.nan
            continue
        arr = np.array([x[1] for x in ics])
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        icir = mu / sd if sd > 0 else 0.0
        hit = float((arr > 0).mean()) if mu >= 0 else float((arr < 0).mean())
        decay[h] = mu
        if h == admission_horizon:
            ics_main = ics
    if not ics_main:
        raise ValueError("no IC observations at admission horizon")
    arr = np.array([x[1] for x in ics_main])
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    icir = mu / sd if sd > 0 else 0.0
    hit = float((arr > 0).mean()) if mu >= 0 else float((arr < 0).mean())
    n_ge8 = sum(1 for _, _ in ics_main)
    # coverage
    total = int(fdf.notna().sum().sum())
    cells = int(fdf.size)
    cov_ad = total / cells if cells else 0.0
    ge8 = int((fdf.notna().sum(axis=1) >= 8).sum())
    cov_d8 = ge8 / len(fdf) if len(fdf) else 0.0
    # turnover: mean absolute rank change at 10d spacing
    sub = fdf.dropna(how="all")
    rows = sub.iloc[::10]
    ranks = rows.rank(axis=1)
    chg = []
    prev = None
    for _, r in ranks.iterrows():
        r = r.dropna()
        if prev is not None:
            both = prev.index.intersection(r.index)
            if len(both) >= min_assets:
                chg.append(float((r[both] - prev[both]).abs().mean()))
        prev = r
    to = float(np.mean(chg)) if chg else float("nan")
    return {
        "ic": round(mu, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 4),
        "n_ic_dates": len(ics_main), "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov_d8, 4), "turnover_10d_rank": round(to, 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in horizons},
        "panel": fdf,
    }


def load_live_library():
    lib = {}
    for p in sorted(glob.glob("factors/*.json")):
        base = os.path.basename(p)
        if base.endswith(".bak") or base == "factor_ensemble.json":
            continue
        try:
            d = json.load(open(p))
            if d.get("validation", {}).get("status") != "EFFECTIVE":
                continue
            art = d["validation"].get("signal_artifact")
            if not art:
                continue
            raw = base64.b64decode(art["data"])
            panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                                index_col=0, parse_dates=True)
            panel.index = pd.DatetimeIndex(panel.index)
            lib[d["factor_id"]] = panel
        except Exception as e:
            print(f"  [warn] library load skip {p}: {e}")
    return lib


def library_corrs(panel, lib):
    """Pooled Spearman / pooled Pearson / daily-mean Spearman vs each library member."""
    out = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            out[fid] = {"pooled_spearman": None, "pooled_pearson": None,
                        "daily_spearman": None, "n_pairs": 0}
            continue
        x = panel.loc[common, cols].values.ravel()
        y = lp.loc[common, cols].values.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200:
            out[fid] = {"pooled_spearman": None, "pooled_pearson": None,
                        "daily_spearman": None, "n_pairs": int(m.sum())}
            continue
        ps, _ = spearmanr(x[m], y[m])
        pp = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else np.nan
        dw = []
        for dt in common:
            a = panel.loc[dt, cols]
            b = lp.loc[dt, cols]
            mm = a.notna() & b.notna()
            if mm.sum() >= 5:
                v, _ = spearmanr(a[mm], b[mm])
                if np.isfinite(v):
                    dw.append(v)
        ds = float(np.mean(dw)) if dw else np.nan
        out[fid] = {"pooled_spearman": round(float(ps), 4),
                    "pooled_pearson": round(float(pp), 4),
                    "daily_spearman": round(float(ds), 4) if np.isfinite(ds) else None,
                    "n_pairs": int(m.sum())}
    return out


def max_abs_library_corr(panel, lib):
    cs = library_corrs(panel, lib)
    vals = [abs(v["pooled_spearman"]) for v in cs.values()
            if v.get("pooled_spearman") is not None]
    return (round(float(max(vals)), 4) if vals else None, cs)


def regime_ic(panel, close):
    fwd = close.pct_change(10).shift(-10)
    out = {}
    for label, lo, hi in [("2020-2021 COVID/recovery", "2020-01-02", "2021-12-31"),
                          ("2022-2023 tightening/AI", "2022-01-01", "2023-12-31"),
                          ("2024-2026-09", "2024-01-01", "2026-09-09"),
                          ("recent3m", "2026-06-09", "2026-09-09")]:
        sub = panel.loc[lo:hi]
        rr = fwd.loc[lo:hi]
        ics = []
        for dt in sub.index:
            fv = sub.loc[dt]
            rv = rr.loc[dt]
            m = fv.notna() & rv.notna()
            if m.sum() >= MIN_ASSETS:
                ic, _ = spearmanr(fv[m], rv[m])
                if np.isfinite(ic):
                    ics.append(ic)
        if ics:
            s = pd.Series(ics)
            out[label] = [round(float(s.mean()), 4),
                          round(float(s.mean() / s.std()), 4) if len(s) > 2 else None,
                          int(len(s))]
        else:
            out[label] = None
    return out


def artifact_b64(panel):
    csv_text = panel.to_csv()
    return base64.b64encode(zlib.compress(csv_text.encode())).decode()


def persist(record, panel):
    v = record.setdefault("validation", {})
    v["signal_artifact"] = {
        "format": "base64:zlib:csv",
        "descrip": "factor value panel rows=date cols=asset (15-asset cross-asset universe)",
        "data": artifact_b64(panel),
    }
    m = v["metrics"]
    record["expected_direction"] = 1 if m["ic"] >= 0 else -1
    record["benchmark_admission"] = {
        "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                     "correlation_threshold": RHO_GATE, "library_capacity": 30,
                     "active_top_k": 10},
        "selected_metrics": {
            "ic": m["ic"], "icir": m["icir"], "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m.get("max_abs_library_correlation"),
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": pd.Timestamp.now().isoformat(),
    }
    path = f"factors/{record['factor_id']}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    with open(path) as f:
        back = json.load(f)
    assert back["factor_id"] == record["factor_id"]
    assert back["validation"]["status"] == "EFFECTIVE"
    assert "signal_artifact" in back["validation"]
    assert abs(back["validation"]["metrics"]["ic"]) >= IC_GATE
    assert abs(back["validation"]["metrics"]["icir"]) >= ICIR_GATE
    print(f"[persist] {path} status={back['validation']['status']} "
          f"art_len={len(back['validation']['signal_artifact']['data'])} reload_ok=True")
    return path