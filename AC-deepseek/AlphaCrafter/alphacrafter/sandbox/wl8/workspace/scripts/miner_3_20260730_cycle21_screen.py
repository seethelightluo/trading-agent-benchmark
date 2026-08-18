"""miner_3 cycle-21: screen novel candidate factors (2026-07-30 sim date).

Library anchors (4 live): usdcny_beta_60, vix_beta_60, eff_ratio_20, downside_ratio_20.
Admission gates: |IC| >= 0.0070 AND |ICIR| >= 0.0840 at h=10; ravel |rho| < 0.5 vs every
library factor and among admissions (mutual). Validation window 2020-01-01..2026-07-29.

Candidate batch (novel / not persisted as EFFECTIVE before):
  sharpe_60, sortino_60, omega_20        : risk-adjusted return / reward-risk quality
  park_vol_20                             : Parkinson vol (high-low based)
  idio_vol_60                             : idiosyncratic vol vs equal-weight market
  vratio_5x60                             : vol acceleration (5d rv / 60d rv)
  ret_skew_20                             : 20d return skewness
  resid_mom_60                            : 60d idiosyncratic momentum (residual sum)
  mom_60_skip5                            : 60d momentum skipping 5d
  gold_beta_60                            : 60d beta to XAU returns
  hl_range_20                             : 20d mean (H-L)/C range ratio
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
LIB_FIDS = ["usdcny_beta_60", "vix_beta_60", "eff_ratio_20", "downside_ratio_20"]
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
    highs = {a: d["high"].astype(float) for a, d in data.items()}
    lows = {a: d["low"].astype(float) for a, d in data.items()}
    cands = {}

    def rv_window(a, n):
        return rets[a].rolling(n).std() * np.sqrt(252)

    # sharpe_60: 60d mean/std of daily returns
    cands["sharpe_60"] = {a: r.rolling(60).mean() / r.rolling(60).std() for a, r in rets.items()}
    # sortino_60: 60d mean / downside semi-dev
    so = {}
    for a, r in rets.items():
        dd = r.clip(upper=0.0) ** 2
        so[a] = r.rolling(60).mean() / np.sqrt(dd.rolling(60).mean())
    cands["sortino_60"] = so
    # omega_20: sum gains / |sum losses| over 20d
    om = {}
    for a, r in rets.items():
        g = r.clip(lower=0.0).rolling(20).sum()
        l = (-r.clip(upper=0.0)).rolling(20).sum()
        om[a] = g / l.replace(0, np.nan)
    cands["omega_20"] = om
    # park_vol_20: Parkinson 20d vol from high/low
    pk = {}
    for a in closes:
        hl = (highs[a] / lows[a]).apply(np.log) ** 2
        pk[a] = np.sqrt(hl.rolling(20).mean() / (4 * np.log(2)))
    cands["park_vol_20"] = pk
    # idio_vol_60: residual std vs equal-weight market (60d)
    rdf = pd.DataFrame(rets)
    mkt = rdf.mean(axis=1)
    iv = {}
    for a in rdf:
        j = pd.concat([rdf[a].rename("y"), mkt.rename("x")], axis=1).dropna()
        def resid_std(n):
            q = j.tail(n)
            var = float(q.x.var())
            if len(q) < 12 or var <= 1e-14:
                return float("nan")
            b = float(q.y.cov(q.x) / var)
            return float((q.y - b * q.x).std())
        iv[a] = j["y"].rolling(60).apply(lambda w: resid_std(60), raw=False)
    cands["idio_vol_60"] = iv
    # vratio_5x60: 5d realized vol / 60d realized vol
    cands["vratio_5x60"] = {a: rv_window(a, 5) / rv_window(a, 60) for a in rets}
    # ret_skew_20: 20d skewness of returns
    cands["ret_skew_20"] = {a: r.rolling(20).skew() for a, r in rets.items()}
    # resid_mom_60: 60d cumulative idiosyncratic return
    rm = {}
    for a in rdf:
        j = pd.concat([rdf[a].rename("y"), mkt.rename("x")], axis=1).dropna()
        def resid_sum(n):
            q = j.tail(n)
            var = float(q.x.var())
            if len(q) < 12 or var <= 1e-14:
                return float("nan")
            b = float(q.y.cov(q.x) / var)
            return float((q.y - b * q.x).sum())
        rm[a] = j["y"].rolling(60).apply(lambda w: resid_sum(60), raw=False)
    cands["resid_mom_60"] = rm
    # mom_60_skip5
    cands["mom_60_skip5"] = {a: c / c.shift(65) - 1.0 for a, c in closes.items()}
    # gold_beta_60: beta of asset returns to XAU returns
    xau = rets["XAU"]
    gb = {}
    for a, r in rets.items():
        m = xau.reindex(r.index).ffill()
        gb[a] = r.rolling(60).cov(m) / m.rolling(60).var().replace(0, np.nan)
    cands["gold_beta_60"] = gb
    # hl_range_20: 20d mean of (H-L)/C
    cands["hl_range_20"] = {a: ((highs[a] - lows[a]) / closes[a]).rolling(20).mean() for a in closes}
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


REG_WINDOWS = {"2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
               "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
               "2024-2026-07 crypto/commodity": ("2024-01-01", None)}


def regime_ic(panel, data):
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


def main():
    data = load_data()
    macro = load_macro()
    lib = load_lib_panels()
    print(f"[load] {len(data)}/15 assets; macro={len(macro)}; lib={list(lib)}; end={END.date()}", flush=True)
    cands = make_candidates(data, macro)
    cands = {fid: pd.DataFrame(p) for fid, p in cands.items()}

    results = {}
    for fid, panel in cands.items():
        tbl = ic_table(panel, data)
        prim = tbl[10]
        if prim is None:
            print(f"{fid:14s} NO IC DATA")
            continue
        ic, icir = prim["ic"], prim["icir"]
        cov, to = coverage(panel, data), rank_turnover(panel)
        dec = {str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}
        reg = regime_ic(panel, data)
        rho_lib = {}
        for fid2, lp in lib.items():
            r, ns = ravel_rho(panel, lp)
            if np.isfinite(r):
                rho_lib[fid2] = round(float(r), 4)
        max_lib = max([abs(v) for v in rho_lib.values()], default=0.0)
        gate_ic = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
        gate_rho = max_lib < RHO_GATE
        flag = "PASS" if gate_ic and gate_rho else "fail"
        results[fid] = dict(ic=ic, icir=icir, hit=prim["hit"], n=prim["n"], cov=cov, to=to,
                            decay=dec, regime=reg, rho_lib=rho_lib, max_lib=max_lib, flag=flag)
        print(f"{fid:14s} IC={ic:+.4f} ICIR={icir:+.4f} hit={prim['hit']:.3f} n={prim['n']} "
              f"cov={cov:.3f} TO={to:.2f} | max_rho_lib={max_lib:.3f} | {flag}")
    print("\n=== detail ===")
    for fid, r in results.items():
        print(f"\n{fid} [{r['flag']}]")
        print(f"  decay={r['decay']}")
        print(f"  regime={r['regime']}")
        print(f"  rho_lib={r['rho_lib']}")
    with open("scripts/_miner3_cycle21_screen_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()