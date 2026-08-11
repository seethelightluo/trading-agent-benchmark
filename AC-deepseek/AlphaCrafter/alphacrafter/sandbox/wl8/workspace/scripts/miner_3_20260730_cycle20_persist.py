"""miner_3 cycle-20: persist cycle-19 deep-validated PASS candidates (2026-07-30).

Admissions (h=10 admission horizon, validation window 2020-01-01..2026-07-29):
  1. vix_beta_60       : 60d beta of asset returns to VIX pct change   (IC<0 -> dir -1)
  2. eff_ratio_20      : Kaufman efficiency ratio 20d                  (IC>0 -> dir +1)
  3. downside_ratio_20 : 20d downside semi-deviation / total vol       (IC<0 -> dir -1)

Crowding contract: zscore_60 has |rho|=0.741 vs downside_ratio_20 (>=0.5) and lower
quality (q=0.00636 < 0.00668), so only downside_ratio_20 is admitted from that pair.
Rho vs live library anchor (usdcny_beta_60) recomputed gate-style (ravel spearman).
"""
import json, base64, zlib, io, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
IC_GATE, ICIR_GATE, RHO_GATE = 0.0070, 0.0840, 0.5
MIN_ASSETS = 8
END = pd.Timestamp("2026-07-29")
LIB_FIDS = ["usdcny_beta_60"]  # current live library

t0 = time.time()

def load_data():
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[s] = df
    return out

def load_macro():
    m = {}
    for name in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{name}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        m[name] = df["close"].astype(float)
    return m

def load_lib_panels():
    lib = {}
    for fid in LIB_FIDS:
        d = json.load(open(f"factors/{fid}.json"))
        raw = zlib.decompress(base64.b64decode(d["validation"]["signal_artifact"]["data"])).decode("utf-8")
        panel = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib

def make_candidates(data, macro):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    rets = {a: c.pct_change() for a, c in closes.items()}
    cands = {}
    # vix_beta_60
    vix = macro["VIX"].pct_change()
    vb = {}
    for a, r in rets.items():
        m = vix.reindex(r.index).ffill()
        vb[a] = r.rolling(60).cov(m) / m.rolling(60).var().replace(0, np.nan)
    cands["vix_beta_60"] = vb
    # eff_ratio_20 (Kaufman efficiency)
    cands["eff_ratio_20"] = {a: (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
                             for a, c in closes.items()}
    # downside_ratio_20
    dr = {}
    for a, r in rets.items():
        neg2 = r.clip(upper=0.0) ** 2
        dr[a] = np.sqrt(neg2.rolling(20).mean()) / r.rolling(20).std()
    cands["downside_ratio_20"] = dr
    return cands

def ic_table(factor, data, horizons=(1, 2, 3, 5, 10, 20)):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    fdf = pd.DataFrame(factor)
    res = {}
    for h in horizons:
        fwd = pd.DataFrame({a: c.shift(-h) / c - 1.0 for a, c in closes.items()})
        common = fdf.index.intersection(fwd.index)
        ics, ge8 = [], 0
        fv, rv = fdf.loc[common].values, fwd.loc[common].values
        for i in range(len(common)):
            fr, rr = fv[i], rv[i]
            m = ~(np.isnan(fr) | np.isnan(rr))
            if m.sum() < MIN_ASSETS:
                continue
            ic = spearmanr(fr[m], rr[m])[0]
            if np.isfinite(ic):
                ics.append(ic)
                if m.sum() >= 8:
                    ge8 += 1
        if not ics:
            res[h] = None
            continue
        a = np.array(ics)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        res[h] = dict(ic=float(a.mean()), icir=(float(a.mean() / std) if std > 0 else 0.0),
                      n=len(a), hit=float((a > 0).mean()), ge8_frac=ge8 / len(a))
    return res

def coverage(factor, data):
    tot = val = 0
    for a, s in factor.items():
        if a not in data:
            continue
        tot += len(s)
        val += int(pd.Series(s).dropna().shape[0])
    return val / tot if tot else 0.0

def rank_turnover(factor, step=10):
    fdf = pd.DataFrame(factor).dropna(how="all")
    if len(fdf) < 3 * step:
        return float("nan")
    rows = fdf.iloc[::step].rank(axis=1)
    chg, prev = [], None
    for _, r in rows.iterrows():
        r = r.dropna()
        if prev is not None:
            both = r.index.intersection(prev.index)
            if len(both) >= MIN_ASSETS:
                chg.append(float((r[both] - prev[both]).abs().mean()))
        prev = r
    return float(np.mean(chg)) if chg else float("nan")

def ravel_rho(a_panel, b_panel):
    a = pd.DataFrame(a_panel).stack(); b = pd.DataFrame(b_panel).stack()
    a = a[a.notna()]; b = b[b.notna()]
    both = a.index.intersection(b.index)
    if len(both) < 30:
        return float("nan"), len(both)
    return float(spearmanr(a.loc[both].values, b.loc[both].values)[0]), len(both)

def datewise_mean_abs_rho(a_panel, b_panel, min_per=3):
    cf, af = pd.DataFrame(a_panel), pd.DataFrame(b_panel)
    vals = []
    for dt in cf.index.intersection(af.index):
        x, y = cf.loc[dt].dropna(), af.loc[dt].dropna()
        b = x.index.intersection(y.index)
        if len(b) >= min_per:
            r = spearmanr(x[b], y[b])[0]
            if np.isfinite(r):
                vals.append(abs(r))
    return float(np.mean(vals)) if vals else float("nan")

def artifact_b64(panel):
    return base64.b64encode(zlib.compress(panel.to_csv().encode())).decode()

REG_WINDOWS = {"2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
               "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
               "2024-2026-07 crypto/commodity": ("2024-01-01", None)}

def regime_table(panel, data):
    out = {}
    for rname, (r0, r1) in REG_WINDOWS.items():
        sub = panel.loc[(panel.index >= r0) & (panel.index <= (r1 or END))]
        if len(sub) < 30:
            out[rname] = None
            continue
        st = ic_table(sub, data, horizons=(10,))
        if st[10] and st[10]["n"] >= 15:
            out[rname] = [round(st[10]["ic"], 4), round(st[10]["icir"], 4), st[10]["n"]]
        else:
            out[rname] = None
    return out

def build_record(fid, fname, expr, desc, deps, params, tags, panel, data, direction, regime_notes, lib):
    panel = pd.DataFrame(panel)
    tbl = ic_table(panel, data)
    prim = tbl[10]
    ic, icir = prim["ic"], prim["icir"]
    hit = prim["hit"] if ic >= 0 else 1.0 - prim["hit"]
    cov_ad, to = coverage(panel, data), rank_turnover(panel)
    decay = {str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}
    reg = regime_table(panel, data)
    rho_lib, mutual = {}, {}
    for fid2, lp in lib.items():
        r, ns = ravel_rho(panel, lp)
        if np.isfinite(r):
            rho_lib[fid2] = round(float(r), 4)
    # mutual rho among the three admissions (all vs all, dedup)
    max_mutual = 0.0
    for oid, op in ADMISSIONS_PANELS.items():
        if oid == fid:
            continue
        r, ns = ravel_rho(panel, op)
        if np.isfinite(r):
            mutual[oid] = round(float(r), 4)
            max_mutual = max(max_mutual, abs(r))
    max_lib = max([abs(v) for v in rho_lib.values()], default=0.0)
    assert abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE, f"{fid} IC gate fail"
    assert max_lib < RHO_GATE, f"{fid} library rho gate fail {rho_lib}"
    assert max_mutual < RHO_GATE, f"{fid} mutual rho gate fail {mutual}"
    print(f"\n=== {fid} ===")
    print(f"  IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={prim['n']} ge8={prim['ge8_frac']:.3f} "
          f"cov={cov_ad:.4f} TO={to:.3f}")
    print(f"  decay={decay}")
    print(f"  regime={reg}")
    print(f"  rho_lib={rho_lib} max={max_lib:.3f} mutual={mutual} max_mutual={max_mutual:.3f}")
    rec = dict(
        factor_id=fid, factor_name=fname, version="1.0.0",
        calculation=dict(expression=expr, description=desc),
        dependencies=deps, parameters=params, tags=tags,
        expected_direction=direction,
        validation=dict(
            status="EFFECTIVE", period="2020-01-01..2026-07-29", last_validated="2026-07-30",
            admission_horizon=10, regime_notes=regime_notes,
            metrics=dict(
                ic=round(ic, 4), icir=round(icir, 4), ic_hit_ratio=round(hit, 4),
                n_ic_dates=int(prim["n"]), coverage_asset_days=round(cov_ad, 4),
                coverage_dates_ge8=round(prim["ge8_frac"], 4), turnover_10d_rank=round(to, 4),
                decay_ic_by_horizon=decay, regime_ic_icir=reg,
                max_abs_library_correlation=(round(max_lib, 4) if max_lib else None),
                library_correlation_detail={k: round(v, 4) for k, v in rho_lib.items()},
                max_abs_mutual_rho=round(max_mutual, 4),
                mutual_rho_detail={k: round(v, 4) for k, v in mutual.items()},
            ),
        ),
    )
    rec["validation"]["signal_artifact"] = {
        "format": "base64:zlib:csv",
        "descrip": "factor value panel rows=date cols=asset (15-asset cross-asset universe)",
        "data": artifact_b64(panel),
    }
    m = rec["validation"]["metrics"]
    rec["benchmark_admission"] = {
        "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                     "correlation_threshold": RHO_GATE, "library_capacity": 30, "active_top_k": 10},
        "selected_metrics": {
            "ic": m["ic"], "icir": m["icir"], "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": pd.Timestamp.now().isoformat(),
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    with open(path) as f:
        back = json.load(f)
    assert back["factor_id"] == fid
    assert back["validation"]["status"] == "EFFECTIVE"
    assert "signal_artifact" in back["validation"]
    assert abs(back["validation"]["metrics"]["ic"]) >= IC_GATE
    assert abs(back["validation"]["metrics"]["icir"]) >= ICIR_GATE
    print(f"  [persist] {path} status={back['validation']['status']} "
          f"art_len={len(back['validation']['signal_artifact']['data'])} reload_ok=True")
    return rec

def main():
    data = load_data()
    macro = load_macro()
    lib = load_lib_panels()
    print(f"[load] {len(data)}/15 assets; macro={len(macro)}; lib={list(lib)}; end={END.date()}", flush=True)
    cands = make_candidates(data, macro)
    global ADMISSIONS_PANELS
    ADMISSIONS_PANELS = {fid: pd.DataFrame(p) for fid, p in cands.items()}

    build_record(
        "vix_beta_60", "60d VIX beta",
        "rolling 60d beta of asset daily returns to VIX daily pct change (cov/var)",
        "Sensitivity of each asset's daily returns to VIX moves over the past 60 trading days. "
        "High-VIX-beta assets (risk-sensitive: equities, crypto) tend to underperform over the "
        "next 10 trading days (risk-premium timing / defensive tilt). Negative IC across all "
        "three regimes and every horizon (h=1 -0.020 -> h=20 -0.134).",
        ["close", "VIX"], {"lookback": 60}, ["risk", "beta", "volatility", "cross-asset"],
        cands["vix_beta_60"], data, -1,
        "Validated 2020-01-01..2026-07-29 (n=1624 IC dates, 70.3% asset-day coverage, TO=1.05). "
        "Regime ICs all negative: COVID/recovery -0.204 (ICIR -0.56), 2022-23 -0.041 (-0.094), "
        "2024-26 -0.044 (-0.112). Orthogonal to live library (rho vs usdcny_beta_60 = -0.16).",
        lib)

    build_record(
        "eff_ratio_20", "20d Efficiency Ratio (Kaufman)",
        "abs(close - close.shift(20)) / sum(abs(diff(close)), 20)",
        "Kaufman efficiency ratio over 20d: net displacement divided by total path length "
        "(1.0 = perfectly trending, 0 = choppy). Trend-efficient assets tend to outperform "
        "over the next 10 trading days in this cross-asset universe (trend persistence).",
        ["close"], {"lookback": 20}, ["trend", "efficiency", "momentum", "cross-asset"],
        cands["eff_ratio_20"], data, 1,
        "Validated 2020-01-01..2026-07-29 (n=1665 IC dates, 72.0% asset-day coverage, TO=4.05). "
        "Positive IC at every horizon (h=1 +0.018 -> h=20 +0.044), regime ICs all positive and "
        "improving (2024-26 ICIR +0.20). Re-admission: previously evicted in cycle 9 for crowding "
        "with yield_beta_cond_60x20 (rho 0.55) which is no longer in the library; rho vs current "
        "anchor usdcny_beta_60 = -0.03.",
        lib)

    build_record(
        "downside_ratio_20", "20d Downside Semi-Deviation Ratio",
        "sqrt(mean(min(ret,0)^2, 20)) / std(ret, 20)",
        "Ratio of 20-day downside semi-deviation (root mean square of negative daily returns) to "
        "20-day total volatility. Assets whose recent risk is concentrated on the downside "
        "(high ratio) tend to underperform over the next 10 trading days (downside-risk premium / "
        "tail-risk continuation). Negative IC in all three regimes.",
        ["close"], {"lookback": 20}, ["risk", "downside", "volatility", "cross-asset"],
        cands["downside_ratio_20"], data, -1,
        "Validated 2020-01-01..2026-07-29 (n=1665 IC dates, 72.0% asset-day coverage, TO=3.15). "
        "Negative IC at all horizons (h=1 -0.010 -> h=20 -0.047), regime ICs all negative with "
        "ICIR <= -0.12 in every window. Chosen over zscore_60 (|rho|=0.741, above the 0.5 crowding "
        "contract) on quality q=|IC|*|ICIR|=0.0067 vs 0.0064. Orthogonal to live library "
        "(rho vs usdcny_beta_60 = -0.09).",
        lib)

    print(f"\n[all done] elapsed={time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
