"""miner_3 cycle-18 screening: calibrate pairwise-rho vs the usdcny_beta_60 anchor
panel (reconstructed from its persisted signal artifact), then screen a batch of
diverse factor candidates for IC/ICIR gate + anchor orthogonality (<0.5 Spearman).
Data visible through 2026-07-29 (decision date 2026-07-30 uses prior close).
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

# ---------------- data loading ----------------
def get_watchlist():
    try:
        wl = get_account_dict().get("watch_list") or []
        if wl:
            return list(wl)
    except Exception:
        pass
    return WATCH

def load_data(days=3200):
    out = {}
    for s in get_watchlist():
        try:
            df = get_stock_daily_data(symbol=s, days=days)
        except Exception:
            df = None
        if df is None or len(df) < 400:
            print(f"[load] {s}: insufficient ({0 if df is None else len(df)})")
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[s] = df
    print(f"[load] {len(out)}/15 instruments; {min(d.index.min() for d in out.values()).date()} .. "
          f"{max(d.index.max() for d in out.values()).date()}")
    return out

def load_macro():
    m = {}
    for name in MACRO:
        try:
            df = pd.read_csv(f"../persistent/index_data/{name}.csv")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            m[name] = df["close"].astype(float)
        except Exception as e:
            print(f"[macro] {name} failed: {e}")
    print(f"[macro] loaded {list(m)}")
    return m

def anchor_panel():
    d = json.load(open("factors/usdcny_beta_60.json"))
    art = d["validation"]["signal_artifact"]["data"]
    raw = zlib.decompress(base64.b64decode(art)).decode("utf-8")
    fdf = pd.read_csv(io.StringIO(raw), index_col=0)
    fdf.index = pd.to_datetime(fdf.index)
    return fdf

# ---------------- IC machinery ----------------
def ic_table(factor, data, horizons=(1, 2, 3, 5, 10, 20), min_assets=8, primary_h=10):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    res = {}
    for h in horizons:
        fwd = {a: c.shift(-h) / c - 1.0 for a, c in closes.items()}
        fdf = pd.DataFrame(factor)
        rdf = pd.DataFrame(fwd)
        common = fdf.index.intersection(rdf.index)
        ics, n_ge8 = [], 0
        for dt in common:
            f = fdf.loc[dt].dropna()
            r = rdf.loc[dt].dropna()
            both = f.index.intersection(r.index)
            if len(both) < min_assets:
                continue
            ic, _ = spearmanr(f[both], r[both])
            if np.isfinite(ic):
                ics.append(ic)
                if len(both) >= 8:
                    n_ge8 += 1
        if not ics:
            res[h] = None
            continue
        a = np.array(ics)
        mean_ic = float(a.mean())
        std_ic = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        res[h] = dict(ic=mean_ic, icir=(mean_ic / std_ic if std_ic > 0 else 0.0),
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
    """Raveled Spearman on shared non-null (date, asset) cells + date-wise mean abs."""
    c = pd.DataFrame(cand_panel).stack()
    a = anchor.stack()
    c = c[c.notna()]; a = a[a.notna()]
    both = c.index.intersection(a.index)
    if len(both) < 30:
        return float("nan"), float("nan"), len(both)
    rho, _ = spearmanr(c.loc[both].values, a.loc[both].values)
    # date-wise
    cf = pd.DataFrame(cand_panel); af = anchor
    dts = cf.index.intersection(af.index)
    dw = []
    for dt in dts:
        x = cf.loc[dt].dropna(); y = af.loc[dt].dropna()
        b = x.index.intersection(y.index)
        if len(b) >= 3:
            r, _ = spearmanr(x[b], y[b])
            if np.isfinite(r):
                dw.append(r)
    return float(rho), (float(np.mean(np.abs(dw))) if dw else float("nan")), len(both)

# ---------------- candidates ----------------
def make_candidates(data, macro):
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    rets = {a: c.pct_change() for a, c in closes.items()}
    vols = {a: r.rolling(20).std() for a, r in rets.items()}
    cands = {}

    def reg_ret(series, lag):
        return series / series.shift(lag) - 1.0

    def beta_to(asset_rets, mkt_rets, win):
        return {a: asset_rets[a].rolling(win).cov(mkt_rets) / mkt_rets.rolling(win).var()
                for a in asset_rets}

    # 1) DXY beta 60
    cands["dxy_beta_60"] = beta_to(rets, macro["DXY"].pct_change(), 60)
    # 2) USDJPY beta 60
    cands["usdjpy_beta_60"] = beta_to(rets, macro["USDJPY"].pct_change(), 60)
    # 3) EURUSD beta 60
    cands["eurusd_beta_60"] = beta_to(rets, macro["EURUSD"].pct_change(), 60)
    # 4) VIX beta 60 (unconditional)
    cands["vix_beta_60"] = beta_to(rets, macro["VIX"].pct_change(), 60)
    # 5) US10Y beta 60 (yield change sensitivity)
    us10y_chg = closes["US10Y"].diff()
    cands["us10y_beta_60"] = beta_to(rets, us10y_chg, 60)
    # 6) serial correlation 10d
    cands["serial_corr_10"] = {a: r.rolling(10).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False) for a, r in rets.items()}
    # 7) downside vol ratio 20
    def downside_ratio(r, win=20):
        neg = r.clip(upper=0.0)
        dvol = (neg ** 2).rolling(win).mean().apply(np.sqrt)
        tvol = r.rolling(win).std()
        return dvol / tvol
    cands["downside_ratio_20"] = {a: downside_ratio(r) for a, r in rets.items()}
    # 8) vol percentile 60 (rank of |ret| 20d avg within trailing 60d)
    def vol_pct(r, win=60, sub=20):
        v = (r.abs()).rolling(sub).mean()
        return v.rolling(win).apply(lambda x: (x[-1] >= x).mean() if len(x) == win else np.nan, raw=True)
    cands["vol_pct_60"] = {a: vol_pct(r) for a, r in rets.items()}
    # 9) vol ratio 5x20
    v5 = {a: r.rolling(5).std() for a, r in rets.items()}
    cands["vol_ratio_5x20"] = {a: v5[a] / vols[a] for a in rets}
    # 10) range ratio 5x20 (high-low)/close
    def range_ratio(data, win):
        out = {}
        for a, d in data.items():
            rr = (d["high"] - d["low"]) / d["close"]
            out[a] = rr.rolling(win).mean()
        return out
    rr5, rr20 = range_ratio(data, 5), range_ratio(data, 20)
    cands["range_ratio_5x20"] = {a: rr5[a] / rr20[a] for a in rr5}
    # 11) volume z 5x20 (volume trend)
    v20 = {a: d["volume"].rolling(20).mean() for a, d in data.items()}
    v60 = {a: d["volume"].rolling(60).mean() for a, d in data.items()}
    vstd60 = {a: d["volume"].rolling(60).std() for a, d in data.items()}
    cands["volume_z_5x60"] = {a: (v20[a] - v60[a]) / vstd60[a].replace(0, np.nan) for a in data}
    # 12) overnight gap mean 20
    def gap_mean(d, win=20):
        g = d["open"] / d["close"].shift(1) - 1.0
        return g.rolling(win).mean()
    cands["gap_mean_20"] = {a: gap_mean(d) for a, d in data.items()}
    # 13) skew 20
    cands["skew_20"] = {a: r.rolling(20).skew() for a, r in rets.items()}
    # 14) kurt 60
    cands["kurt_60"] = {a: r.rolling(60).kurt() for a, r in rets.items()}
    # 15) efficiency ratio 20
    def eff_ratio(c, win=20):
        move = (c - c.shift(win)).abs()
        path = c.diff().abs().rolling(win).sum()
        return move / path
    cands["eff_ratio_20"] = {a: eff_ratio(c) for a, c in closes.items()}
    # 16) zscore_60 (mean reversion distance)
    ma60 = {a: c.rolling(60).mean() for a, c in closes.items()}
    sd60 = {a: c.rolling(60).std() for a, c in closes.items()}
    cands["zscore_60"] = {a: (c - ma60[a]) / sd60[a].replace(0, np.nan) for a, c in closes.items()}
    # 17) hl_pos_20
    cands["hl_pos_20"] = {a: ((d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)).rolling(20).mean()
                          for a, d in data.items()}
    # 18) cny_beta_60 replicate (calibration, expect high rho)
    cands["cny_beta_60_rep"] = beta_to(rets, macro["USDCNY"].pct_change(), 60)
    return cands

def main():
    data = load_data()
    macro = load_macro()
    anchor = anchor_panel()
    print(f"[anchor] panel {anchor.shape}, non-null {int(anchor.notna().sum().sum())}")

    cands = make_candidates(data, macro)
    rows = []
    for fid, panel in cands.items():
        tbl = ic_table(panel, data)
        prim = tbl.get(10)
        if prim is None:
            print(f"{fid:22s} DEGENERATE (no IC dates)")
            continue
        rho_r, rho_dw, nshared = rho_vs_anchor(panel, anchor)
        cov = coverage(panel, data)
        to = rank_turnover(panel)
        gate_ic = abs(prim["ic"]) >= IC_GATE and abs(prim["icir"]) >= ICIR_GATE
        gate_rho = (np.isfinite(rho_r) and abs(rho_r) < RHO_GATE)
        flag = "PASS" if (gate_ic and gate_rho) else ("IC-OK-RHO-HI" if (gate_ic and not gate_rho) else "fail")
        print(f"{fid:22s} IC={prim['ic']:+.4f} ICIR={prim['icir']:+.4f} hit={prim['hit']:.3f} "
              f"n={prim['n']:5d} cov={cov:.3f} to={to:.2f} rho_ravel={rho_r:+.3f} rho_dw={rho_dw:+.3f} "
              f"nshared={nshared} -> {flag}")
        decay = {str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}
        rows.append(dict(fid=fid, ic=prim["ic"], icir=prim["icir"], hit=prim["hit"], n=prim["n"],
                         cov=cov, to=to, rho_r=rho_r, rho_dw=rho_dw, nshared=nshared,
                         decay=decay, flag=flag))
    print("\n===== SUMMARY =====")
    for r in sorted(rows, key=lambda x: (x["flag"] != "PASS", -abs(x["ic"]))):
        print(f"{r['fid']:22s} {r['flag']:12s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} "
              f"rho={r['rho_r']:+.3f} nshared={r['nshared']} decay10={r['decay'].get('10')}")
    with open("scripts/_miner3_cycle18_screen_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print("[saved] scripts/_miner3_cycle18_screen_results.json")

if __name__ == "__main__":
    main()
