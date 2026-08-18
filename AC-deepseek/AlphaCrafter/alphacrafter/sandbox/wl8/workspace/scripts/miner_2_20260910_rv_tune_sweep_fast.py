"""miner_2 2026-09-10: fast vectorized tune of risk-adjusted reversal family.

Vectorized rank-IC (Spearman via Pearson on ranks) to compute per-date IC without
per-date scipy overhead. Same gates: |IC|>=0.0070, |ICIR|>=0.0840, min 8 assets.
END = 2026-09-09 (visible through decision date 2026-09-10).
"""
import base64, io, json, zlib
import numpy as np
import pandas as pd

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
END = pd.Timestamp("2026-09-09")
START = pd.Timestamp("2021-01-01")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS = 8


def load_panel():
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


def fwd_returns(horizon):
    out = {}
    for a in ASSETS:
        c = DENSE[a]["close"]
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)


FWD = {h: fwd_returns(h) for h in [1, 2, 3, 5, 10, 20]}


def ic_series_ranked(factor_df, fwd_df):
    """Per-date Spearman IC via Pearson on cross-sectional ranks."""
    fr = fwd_df.rank(axis=1)
    fr_vals = fr.values
    idx = factor_df.index
    dates, ics = [], []
    fv = factor_df.values
    for i, dt in enumerate(idx):
        x = fv[i]
        m = np.isfinite(x)
        yr = fr_vals[i]
        m &= np.isfinite(yr)
        if m.sum() < MIN_ASSETS:
            continue
        # rank factor values (already need ranking per date)
        xr = np.full(x.shape, np.nan)
        xr[m] = pd.Series(x[m]).rank().values
        xc = xr - np.nanmean(xr)
        yc = yr - np.nanmean(yr)
        denom = np.sqrt(np.nansum(xc * xc) * np.nansum(yc * yc))
        if denom > 0:
            ics.append(np.nansum(xc * yc) / denom)
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize(name, panel):
    ic = ic_series_ranked(panel, FWD[10])
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
    to = float(panel.rank(axis=1).diff(10).abs().mean(axis=1).mean())
    cov_ad = float(panel.notna().sum().sum() / (panel.shape[0] * panel.shape[1]))
    cov_ge8 = float((panel.notna().sum(axis=1) >= MIN_ASSETS).mean())
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        s = ic_series_ranked(panel, FWD[hh])
        s = s[s.index >= START]
        decay[hh] = float(s.mean()) if len(s) else np.nan
    print(f"{name}: " + " | ".join(
        f"{lab} IC={ic_:.4f} ICIR={icir_:.4f} n={n} hit={hit:.3f}"
        for lab, ic_, icir_, n, hit in rows)
          + f" | TO10={to:.2f} cov_ad={cov_ad:.3f} cov_ge8={cov_ge8:.3f}")
    print(f"    decay: { {k: round(v, 4) for k, v in decay.items()} }")
    return {"ic_full": rows[0][1], "icir_full": rows[0][2], "n": rows[0][3],
            "hit": rows[0][4], "turnover_10d": to, "cov_ad": cov_ad,
            "cov_ge8": cov_ge8, "decay": decay}


def factor_panel_std(w_ret, w_vol, skip=0, use_mean_ret=False):
    out = {}
    for a in ASSETS:
        c = DENSE[a]["close"]
        r = c.pct_change()
        wret = r.rolling(w_ret).sum()
        v = r.rolling(w_vol).std()
        if skip > 0:
            wret = wret - r.rolling(skip).sum()
        if use_mean_ret:
            mret = r.rolling(w_ret, min_periods=w_ret // 2).mean()
            f = -mret / v
        else:
            f = -wret / v
        out[a] = pd.Series(np.asarray(f, dtype=float), index=c.index).reindex(close.index)
    return pd.DataFrame(out)


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


def pooled_rho(p1, p2):
    common = p1.index.intersection(p2.index)
    cols = [c for c in p1.columns if c in p2.columns]
    a = p1.loc[common, cols].values.ravel()
    b = p2.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 200 else np.nan


# ---- sweep ----
panels = {}
results = []
print("\n=== rv_ret family sweep (factor = -w_ret_sum / w_vol_std) ===")
for wr in [5, 10, 15, 20, 30]:
    for wv in [20, 40, 60]:
        name = f"rv_{wr}_{wv}"
        p = factor_panel_std(wr, wv)
        panels[name] = p
        r = summarize(name, p)
        r["name"] = name
        r["rho_lib"] = max_lib_corr(p)
        r["max_abs_library_correlation"] = max([abs(v) for v in r["rho_lib"].values()], default=0.0)
        r["pass"] = abs(r["ic_full"]) >= IC_GATE and abs(r["icir_full"]) >= ICIR_GATE
        results.append(r)

print("\n=== gate summary (|IC|>=0.0070 & |ICIR|>=0.0840, full window) ===")
for r in sorted(results, key=lambda x: -abs(x["icir_full"])):
    print(f"{r['name']}: PASS={r['pass']} IC={r['ic_full']:.4f} ICIR={r['icir_full']:.4f} "
          f"n={r['n']} hit={r['hit']:.3f} TO={r['turnover_10d']:.2f} "
          f"rho_lib={r['rho_lib']}")

print("\n=== sibling pooled correlations (passers only) ===")
passers = [r["name"] for r in results if r["pass"]]
for i, n1 in enumerate(passers):
    for n2 in passers[i + 1:]:
        print(f"  {n1} vs {n2}: {pooled_rho(panels[n1], panels[n2]):.3f}")

print("\n=== variants on the best passer ===")
best = max((r for r in results if r["pass"]), key=lambda x: abs(x["icir_full"]))
best_name = best["name"]
wr, wv = int(best_name.split("_")[1]), int(best_name.split("_")[2])
print(f"Base best: {best_name} wr={wr} wv={wv} (IC={best['ic_full']:.4f} ICIR={best['icir_full']:.4f})")
for skip in [1, 2, 3, 5]:
    name = f"rv_{wr}_{wv}_skip{skip}"
    p = factor_panel_std(wr, wv, skip=skip)
    r = summarize(name, p)
    r["name"] = name
    r["rho_lib"] = max_lib_corr(p)
    r["max_abs_library_correlation"] = max([abs(v) for v in r["rho_lib"].values()], default=0.0)
    r["pass"] = abs(r["ic_full"]) >= IC_GATE and abs(r["icir_full"]) >= ICIR_GATE
    print(f"  {name}: PASS={r['pass']} IC={r['ic_full']:.4f} ICIR={r['icir_full']:.4f} "
          f"n={r['n']} hit={r['hit']:.3f} rho_lib={r['rho_lib']}")

name_m = f"rv_{wr}_{wv}_meanret"
p = factor_panel_std(wr, wv, use_mean_ret=True)
r = summarize(name_m, p)
r["name"] = name_m
r["rho_lib"] = max_lib_corr(p)
r["max_abs_library_correlation"] = max([abs(v) for v in r["rho_lib"].values()], default=0.0)
r["pass"] = abs(r["ic_full"]) >= IC_GATE and abs(r["icir_full"]) >= ICIR_GATE
print(f"  {name_m}: PASS={r['pass']} IC={r['ic_full']:.4f} ICIR={r['icir_full']:.4f} "
      f"n={r['n']} hit={r['hit']:.3f} rho_lib={r['rho_lib']}")