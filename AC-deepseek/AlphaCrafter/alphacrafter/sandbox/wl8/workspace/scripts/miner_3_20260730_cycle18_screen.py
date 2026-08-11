"""miner_3 cycle-18 screening v2 (fast): calibrate pairwise-rho vs usdcny_beta_60
anchor panel and screen candidate batch. Vectorized rolling ops only.
"""
import json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
IC_GATE, ICIR_GATE, RHO_GATE = 0.0070, 0.0840, 0.5

def get_watchlist():
    try:
        wl = get_account_dict().get("watch_list") or []
        if wl:
            return list(wl)
    except Exception:
        pass
    return WATCH

def load_data(days=2400):
    out = {}
    for s in get_watchlist():
        try:
            df = get_stock_daily_data(symbol=s, days=days)
        except Exception:
            df = None
        if df is None or len(df) < 400:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[s] = df
    return out

def load_macro():
    m = {}
    for name in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{name}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        m[name] = df["close"].astype(float)
    return m

def anchor_panel():
    d = json.load(open("factors/usdcny_beta_60.json"))
    raw = zlib.decompress(base64.b64decode(d["validation"]["signal_artifact"]["data"])).decode("utf-8")
    fdf = pd.read_csv(io.StringIO(raw), index_col=0)
    fdf.index = pd.to_datetime(fdf.index)
    return fdf

def ic_table(factor, data, horizons=(1, 2, 3, 5, 10, 20), min_assets=8):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    fdf = pd.DataFrame(factor)
    res = {}
    for h in horizons:
        fwd = pd.DataFrame({a: c.shift(-h) / c - 1.0 for a, c in closes.items()})
        common = fdf.index.intersection(fwd.index)
        ics, n_ge8 = [], 0
        fv = fdf.loc[common].values
        rv = fwd.loc[common].values
        for i in range(len(common)):
            frow, rrow = fv[i], rv[i]
            m = ~(np.isnan(frow) | np.isnan(rrow))
            if m.sum() < min_assets:
                continue
            ic = spearmanr(frow[m], rrow[m])[0]
            if np.isfinite(ic):
                ics.append(ic)
                if m.sum() >= 8:
                    n_ge8 += 1
        if not ics:
            res[h] = None
            continue
        a = np.array(ics)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        res[h] = dict(ic=float(a.mean()), icir=(float(a.mean() / std) if std > 0 else 0.0),
                      n=len(a), hit=float((a > 0).mean()), dates_ge8=n_ge8 / len(a))
    return res

def coverage(factor, data):
    tot = val = 0
    for a, s in factor.items():
        if a not in data:
            continue
        tot += len(s); val += int(s.dropna().shape[0])
    return val / tot if tot else 0.0

def rank_turnover(factor, step=10, min_assets=8):
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

def rho_vs_anchor(cand_panel, anchor):
    c = pd.DataFrame(cand_panel).stack()
    a = anchor.stack()
    c = c[c.notna()]; a = a[a.notna()]
    both = c.index.intersection(a.index)
    if len(both) < 30:
        return float("nan"), float("nan"), len(both)
    rho = spearmanr(c.loc[both].values, a.loc[both].values)[0]
    cf, af = pd.DataFrame(cand_panel), anchor
    dw = []
    for dt in cf.index.intersection(af.index):
        x, y = cf.loc[dt].dropna(), af.loc[dt].dropna()
        b = x.index.intersection(y.index)
        if len(b) >= 3:
            r = spearmanr(x[b], y[b])[0]
            if np.isfinite(r):
                dw.append(r)
    return float(rho), (float(np.mean(np.abs(dw))) if dw else float("nan")), len(both)

def make_candidates(data, macro):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    rets = {a: c.pct_change() for a, c in closes.items()}
    vols20 = {a: r.rolling(20).std() for a, r in rets.items()}
    v5 = {a: r.rolling(5).std() for a, r in rets.items()}
    cands = {}

    def beta_to(asset_rets, mkt, win):
        out = {}
        for a, r in asset_rets.items():
            m = mkt.reindex(r.index).ffill()
            cov = r.rolling(win).cov(m)
            var = m.rolling(win).var()
            out[a] = cov / var.replace(0, np.nan)
        return out

    cands["dxy_beta_60"] = beta_to(rets, macro["DXY"].pct_change(), 60)
    cands["usdjpy_beta_60"] = beta_to(rets, macro["USDJPY"].pct_change(), 60)
    cands["eurusd_beta_60"] = beta_to(rets, macro["EURUSD"].pct_change(), 60)
    cands["vix_beta_60"] = beta_to(rets, macro["VIX"].pct_change(), 60)
    cands["us10y_beta_60"] = beta_to(rets, closes["US10Y"].diff(), 60)
    cands["cny_beta_60_rep"] = beta_to(rets, macro["USDCNY"].pct_change(), 60)

    # serial corr lag1 (vectorized rolling cov)
    sc = {}
    for a, r in rets.items():
        r1 = r.shift(1)
        cov = r.rolling(10).cov(r1)
        sd = r.rolling(10).std() * r1.rolling(10).std()
        sc[a] = cov / sd.replace(0, np.nan)
    cands["serial_corr_10"] = sc

    # downside ratio
    dr = {}
    for a, r in rets.items():
        neg2 = r.clip(upper=0.0) ** 2
        dr[a] = np.sqrt(neg2.rolling(20).mean()) / r.rolling(20).std()
    cands["downside_ratio_20"] = dr

    # vol percentile 60 via sliding window
    vp = {}
    for a, r in rets.items():
        v = r.abs().rolling(20).mean().values
        win = 60
        out = np.full(len(v), np.nan)
        if len(v) >= win:
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(v, win)
            cur = v[win - 1:]
            out[win - 1:] = (cur[:, None] >= sw).mean(axis=1)
        vp[a] = pd.Series(out, index=rets[a].index)
    cands["vol_pct_60"] = vp

    cands["vol_ratio_5x20"] = {a: v5[a] / vols20[a] for a in rets}
    rr5 = {a: ((d["high"] - d["low"]) / d["close"]).rolling(5).mean() for a, d in data.items()}
    rr20 = {a: ((d["high"] - d["low"]) / d["close"]).rolling(20).mean() for a, d in data.items()}
    cands["range_ratio_5x20"] = {a: rr5[a] / rr20[a] for a in data}

    v20 = {a: d["volume"].rolling(20).mean() for a, d in data.items()}
    v60 = {a: d["volume"].rolling(60).mean() for a, d in data.items()}
    vs60 = {a: d["volume"].rolling(60).std() for a, d in data.items()}
    cands["volume_z_5x60"] = {a: (v20[a] - v60[a]) / vs60[a].replace(0, np.nan) for a in data}

    cands["gap_mean_20"] = {a: (d["open"] / d["close"].shift(1) - 1.0).rolling(20).mean() for a, d in data.items()}
    cands["skew_20"] = {a: r.rolling(20).skew() for a, r in rets.items()}
    cands["kurt_60"] = {a: r.rolling(60).kurt() for a, r in rets.items()}
    cands["eff_ratio_20"] = {a: (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum() for a, c in closes.items()}
    ma60 = {a: c.rolling(60).mean() for a, c in closes.items()}
    sd60 = {a: c.rolling(60).std() for a, c in closes.items()}
    cands["zscore_60"] = {a: (c - ma60[a]) / sd60[a].replace(0, np.nan) for a, c in closes.items()}
    cands["hl_pos_20"] = {a: ((d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)).rolling(20).mean()
                          for a, d in data.items()}
    return cands

def main():
    data = load_data()
    macro = load_macro()
    anchor = anchor_panel()
    print(f"[load] {len(data)}/15; [anchor] non-null {int(anchor.notna().sum().sum())}", flush=True)
    cands = make_candidates(data, macro)
    rows = []
    for fid, panel in cands.items():
        tbl = ic_table(panel, data)
        prim = tbl.get(10)
        if prim is None:
            print(f"{fid:22s} DEGENERATE", flush=True)
            continue
        rho_r, rho_dw, nshared = rho_vs_anchor(panel, anchor)
        cov, to = coverage(panel, data), rank_turnover(panel)
        gate_ic = abs(prim["ic"]) >= IC_GATE and abs(prim["icir"]) >= ICIR_GATE
        gate_rho = np.isfinite(rho_r) and abs(rho_r) < RHO_GATE
        flag = "PASS" if (gate_ic and gate_rho) else ("IC-OK-RHO-HI" if (gate_ic and not gate_rho) else "fail")
        print(f"{fid:22s} IC={prim['ic']:+.4f} ICIR={prim['icir']:+.4f} hit={prim['hit']:.3f} "
              f"n={prim['n']:5d} cov={cov:.3f} to={to:.2f} rho_r={rho_r:+.3f} rho_dw={rho_dw:+.3f} "
              f"nshared={nshared} -> {flag}", flush=True)
        rows.append(dict(fid=fid, ic=prim["ic"], icir=prim["icir"], hit=prim["hit"], n=prim["n"],
                         cov=cov, to=to, rho_r=rho_r, rho_dw=rho_dw, nshared=nshared,
                         decay={str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}, flag=flag))
    print("===== SUMMARY =====", flush=True)
    for r in sorted(rows, key=lambda x: (x["flag"] != "PASS", -abs(x["ic"]))):
        print(f"{r['fid']:22s} {r['flag']:12s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} "
              f"rho={r['rho_r']:+.3f} nshared={r['nshared']} d10={r['decay'].get('10')}", flush=True)
    with open("scripts/_miner3_cycle18_screen_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)

if __name__ == "__main__":
    main()
