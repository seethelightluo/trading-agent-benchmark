"""miner_3 cycle-19: deep validation of cycle-18 PASS candidates.
Candidates: vix_beta_60, eff_ratio_20, downside_ratio_20, zscore_60.
Deep checks: full-window IC/ICIR/hit/n/cov/TO, decay by horizon, regime-split
IC/ICIR (2020-21 / 2022-23 / 2024+), pairwise crowding vs live anchor
(usdcny_beta_60) and vs each other. Gates: |IC|>=0.007, |ICIR|>=0.084, |rho|<0.5.
Validation window: 2020-01-01 .. 2026-07-29 (sim current date), h=10 admission.
"""
import json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
IC_GATE, ICIR_GATE, RHO_GATE = 0.0070, 0.0840, 0.5
MIN_ASSETS = 8
END = pd.Timestamp("2026-07-29")

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

def anchor_panel():
    d = json.load(open("factors/usdcny_beta_60.json"))
    raw = zlib.decompress(base64.b64decode(d["validation"]["signal_artifact"]["data"])).decode("utf-8")
    fdf = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
    fdf.index = pd.DatetimeIndex(fdf.index)
    return fdf

def make_candidates(data, macro):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    rets = {a: c.pct_change() for a, c in closes.items()}
    cands = {}
    # vix_beta_60
    def beta_to(asset_rets, mkt, win):
        out = {}
        for a, r in asset_rets.items():
            m = mkt.reindex(r.index).ffill()
            cov = r.rolling(win).cov(m)
            var = m.rolling(win).var()
            out[a] = cov / var.replace(0, np.nan)
        return out
    cands["vix_beta_60"] = beta_to(rets, macro["VIX"].pct_change(), 60)
    # eff_ratio_20 (Kaufman efficiency)
    cands["eff_ratio_20"] = {a: (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
                             for a, c in closes.items()}
    # downside_ratio_20
    dr = {}
    for a, r in rets.items():
        neg2 = r.clip(upper=0.0) ** 2
        dr[a] = np.sqrt(neg2.rolling(20).mean()) / r.rolling(20).std()
    cands["downside_ratio_20"] = dr
    # zscore_60
    ma60 = {a: c.rolling(60).mean() for a, c in closes.items()}
    sd60 = {a: c.rolling(60).std() for a, c in closes.items()}
    cands["zscore_60"] = {a: (c - ma60[a]) / sd60[a].replace(0, np.nan) for a, c in closes.items()}
    return cands

def ic_table(factor, data, horizons=(1, 2, 3, 5, 10, 20), min_assets=MIN_ASSETS):
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
            if m.sum() < min_assets:
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

def regime_split(ic_series_index, tbl_h10):
    """Split a per-horizon IC table by regime windows (recompute IC per window)."""
    regs = {"2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
            "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
            "2024-2026-07 crypto/commodity": ("2024-01-01", "2026-07-29")}
    return regs

def coverage(factor, data):
    tot = val = 0
    for a, s in factor.items():
        if a not in data:
            continue
        tot += len(s)
        val += int(pd.Series(s).dropna().shape[0])
    return val / tot if tot else 0.0

def rank_turnover(factor, step=10, min_assets=MIN_ASSETS):
    fdf = pd.DataFrame(factor).dropna(how="all")
    if len(fdf) < 3 * step:
        return float("nan")
    rows = fdf.iloc[::step].rank(axis=1)
    chg, prev = [], None
    for _, r in rows.iterrows():
        r = r.dropna()
        if prev is not None:
            both = r.index.intersection(prev.index)
            if len(both) >= min_assets:
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

def main():
    data = load_data()
    macro = load_macro()
    anchor = anchor_panel()
    print(f"[load] {len(data)}/15 assets; macro {len(macro)}; anchor nn={int(anchor.notna().sum().sum())}", flush=True)
    cands = make_candidates(data, macro)
    reg_windows = {"2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
                   "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
                   "2024-2026-07 crypto/commodity": ("2024-01-01", "2026-07-29")}
    results = {}
    panels = {"usdcny_beta_60": anchor}
    for fid, panel in cands.items():
        panels[fid] = pd.DataFrame(panel)
    for fid in ["vix_beta_60", "eff_ratio_20", "downside_ratio_20", "zscore_60"]:
        panel = panels[fid]
        tbl = ic_table(panel, data)
        prim = tbl[10]
        cov, to = coverage(panel, data), rank_turnover(panel)
        # regime splits
        reg_out = {}
        for rname, (r0, r1) in reg_windows.items():
            sub = panel.loc[(panel.index >= r0) & (panel.index <= r1)]
            if len(sub) < 30:
                reg_out[rname] = None
                continue
            st = ic_table(sub, data, horizons=(10,))
            if st[10] and st[10]["n"] >= 15:
                reg_out[rname] = [round(st[10]["ic"], 4), round(st[10]["icir"], 4), st[10]["n"]]
            else:
                reg_out[rname] = None
        # crowding vs anchor + vs other candidates
        rho_anchor, ns_a = ravel_rho(panel, anchor)
        dw_anchor = datewise_mean_abs_rho(panel, anchor)
        mutual = {}
        max_abs_rho = abs(rho_anchor) if np.isfinite(rho_anchor) else 0.0
        for oid, op in panels.items():
            if oid == fid or oid == "usdcny_beta_60":
                continue
            r, ns = ravel_rho(panel, op)
            if np.isfinite(r):
                mutual[oid] = round(float(r), 4)
                max_abs_rho = max(max_abs_rho, abs(r))
        gate_ic = abs(prim["ic"]) >= IC_GATE and abs(prim["icir"]) >= ICIR_GATE
        gate_rho = np.isfinite(rho_anchor) and abs(rho_anchor) < RHO_GATE
        flag = "PASS" if (gate_ic and gate_rho) else "FAIL"
        res = dict(
            fid=fid,
            ic=round(prim["ic"], 4), icir=round(prim["icir"], 4),
            hit=round(prim["hit"], 4), n_ic_dates=prim["n"], ge8_frac=round(prim["ge8_frac"], 4),
            coverage_asset_days=round(cov, 4), turnover_10d_rank=round(to, 4),
            decay={str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()},
            regime_ic_icir=reg_out,
            rho_vs_anchor_ravel=round(rho_anchor, 4) if np.isfinite(rho_anchor) else None,
            rho_vs_anchor_datewise_mean_abs=round(dw_anchor, 4) if np.isfinite(dw_anchor) else None,
            max_abs_mutual_rho=round(max_abs_rho, 4),
            mutual_rho=mutual,
            gate_ic=bool(gate_ic), gate_rho=bool(gate_rho), flag=flag)
        results[fid] = res
        print(f"\n=== {fid} === {flag}", flush=True)
        print(f"  IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['hit']:.3f} n={res['n_ic_dates']} "
              f"ge8={res['ge8_frac']:.3f} cov={res['coverage_asset_days']:.4f} TO={res['turnover_10d_rank']:.3f}", flush=True)
        print(f"  decay: {res['decay']}", flush=True)
        print(f"  regime: {reg_out}", flush=True)
        print(f"  rho_anchor_ravel={res['rho_vs_anchor_ravel']:+.3f} dw={res['rho_vs_anchor_datewise_mean_abs']:.3f} "
              f"max_mutual={res['max_abs_mutual_rho']:.3f} mutual={mutual}", flush=True)
    with open("scripts/_miner3_cycle19_deepval_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\n===== SUMMARY =====", flush=True)
    for fid, r in results.items():
        print(f"{fid:22s} {r['flag']:4s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} n={r['n_ic_dates']:5d} "
              f"cov={r['coverage_asset_days']:.3f} TO={r['turnover_10d_rank']:.2f} "
              f"rho_anchor={r['rho_vs_anchor_ravel']:+.3f} max_mutual={r['max_abs_mutual_rho']:.3f}", flush=True)

if __name__ == "__main__":
    main()
