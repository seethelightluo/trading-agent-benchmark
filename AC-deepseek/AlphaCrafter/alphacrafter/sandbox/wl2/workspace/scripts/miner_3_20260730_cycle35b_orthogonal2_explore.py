"""miner_3 cycle 35b: more orthogonal axes (post cryptobeta persistence).

New candidates:
  A. avg_corr_60_fixed   : mean pairwise rolling corr with other 14 assets (own-calendar) - systematic-ness
  B. idio_mom_20         : 20d return residualized on market beta (alpha momentum)
  C. range_pos_day_20    : mean((close-low)/(high-low), 20d) - intraday range position
  D. downbeta_xau_60     : downside beta to XAU (hedge-asset tail co-movement)
  E. downbeta_btc_60     : downside beta to BTC (crypto tail co-movement)
  F. lagbeta_ndx_60      : lagged beta to NDX (tech lead-lag)
  G. volcluster_60       : autocorr of |daily ret| over 60d (vol clustering persistence)
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series,
                         forward_returns, compute_ic, validate_factor)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

lib = {}
for npy in sorted(glob.glob("factors/*.signal.npy")):
    jp = npy.replace(".signal.npy", ".json")
    try:
        d = json.load(open(jp))
        if d.get("validation", {}).get("status") != "EFFECTIVE":
            continue
    except Exception:
        continue
    a = np.load(npy)
    if a.shape == (len(panel), len(panel.columns)):
        lib[Path(npy).stem.replace(".signal", "")] = pd.DataFrame(a, index=panel.index, columns=panel.columns)
print(f"active library ({len(lib)}): {sorted(lib)}")


def stacked_spearman(a, b, per_date_rank=False):
    aa = a.rank(axis=1) if per_date_rank else a
    bb = b.rank(axis=1) if per_date_rank else b
    df = pd.concat([aa.stack().rename("x"), bb.stack().rename("y")], axis=1).dropna()
    return float(df["x"].corr(df["y"], method="spearman")) if len(df) >= 30 else 0.0


def library_corr_metrics(cand):
    out = {}
    for fid, sig in lib.items():
        out[fid] = {"raw": round(stacked_spearman(cand, sig), 4),
                    "ranked": round(stacked_spearman(cand, sig, per_date_rank=True), 4)}
    max_raw = max((abs(v["raw"]) for v in out.values()), default=0.0)
    max_ranked = max((abs(v["ranked"]) for v in out.values()), default=0.0)
    return out, round(max_raw, 4), round(max_ranked, 4)


def per_asset_own(func):
    out = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        out[a] = func(s).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def avg_corr_own_cal(w=60, mp=40):
    ret = panel.pct_change()
    out = {}
    for a in TRADABLES:
        idx_a = ret[a].dropna().index
        cols = []
        for b in TRADABLES:
            if b == a:
                continue
            rb = ret[b].reindex(idx_a)
            cols.append(ret[a].loc[idx_a].rolling(w, min_periods=mp).corr(rb))
        out[a] = pd.concat(cols, axis=1).mean(axis=1).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def idio_mom(w=20, bw=60, mp=30):
    mom = panel / panel.shift(w) - 1.0
    mkt_ret = panel.mean(axis=1).pct_change()
    out = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = df["a"].rolling(bw, min_periods=mp).cov(df["m"]) / df["m"].rolling(bw, min_periods=mp).var()
        mkt_mom = (panel.mean(axis=1) / panel.mean(axis=1).shift(w) - 1.0).reindex(b.index)
        out[a] = (mom[a] - b * mkt_mom).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def range_pos_day(w=20, mp=10):
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= panel.index.max()].set_index("date").astype(float)
        pos = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12)
        out[a] = pos.rolling(w, min_periods=mp).mean().reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def down_beta(macro_ret, w=60, minp=15):
    out = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), macro_ret.rename("m")], axis=1).dropna()
        ser = pd.Series(np.nan, index=df.index)
        for i in range(len(df)):
            if i < w - 1:
                continue
            seg = df.iloc[max(0, i - w + 1): i + 1]
            neg = seg[seg["m"] < 0]
            if len(neg) < minp:
                continue
            v = neg["m"].var()
            if v > 0:
                ser.iloc[i] = neg["a"].cov(neg["m"]) / v
        out[a] = ser.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def lag_beta(macro_ret, w=60, mp=30):
    out = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        ar = s.pct_change()
        m_lag = macro_ret.shift(1)
        df = pd.concat([ar.rename("a"), m_lag.rename("m")], axis=1).dropna()
        b = df["a"].rolling(w, min_periods=mp).cov(df["m"]) / df["m"].rolling(w, min_periods=mp).var()
        out[a] = b.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def volcluster(w=60, mp=40):
    def f(s):
        r = s.pct_change().abs()
        return r.rolling(w, min_periods=mp).corr(r.shift(1))
    return per_asset_own(f)


xau_ret = panel["XAU"].dropna().pct_change()
btc_ret = panel["BTC"].dropna().pct_change()
ndx_ret = panel["NDX"].dropna().pct_change()

cands = {
    "avg_corr_60": avg_corr_own_cal(),
    "idio_mom_20": idio_mom(),
    "range_pos_day_20": range_pos_day(),
    "downbeta_xau_60": down_beta(xau_ret),
    "downbeta_btc_60": down_beta(btc_ret),
    "lagbeta_ndx_60": lag_beta(ndx_ret),
    "volcluster_60": volcluster(),
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, F in cands.items():
    m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    lc, max_raw, max_ranked = library_corr_metrics(F)
    m["max_abs_library_correlation"] = max_raw
    m["max_ranked_library_correlation"] = max_ranked
    m["library_pairwise_corr"] = lc
    m["turnover_10d_rank"] = m.pop("turnover_10_rank", None)
    ic_ser = compute_ic(F, fwd_cache[str(ADM_H)]).dropna()
    reg = {}
    for r0, r1, tag in [("2020-01-01", "2021-12-31", "2020-21"),
                        ("2022-01-01", "2022-12-31", "2022"),
                        ("2023-01-01", "2024-12-31", "2023-24"),
                        ("2025-01-01", "2026-07-29", "2025-26")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            reg[tag] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sd) if sd > 0 else 0.0, 4),
                        "n": int(len(sub))}
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        reg["last250"] = {"ic": round(float(last.mean()), 4),
                          "icir": round(float(last.mean() / sd) if sd > 0 else 0.0, 4),
                          "n": int(len(last))}
    p_ic = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    p_corr = max_raw < 0.45 and max_ranked < 0.45
    p = p_ic and p_corr
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxrho_raw={max_raw} maxrho_ranked={max_ranked} "
          f"=> {'PASS' if p else 'FAIL'}")
    top = sorted(lc.items(), key=lambda kv: -abs(kv[1]["raw"]))[:3]
    print(f"    top-rho: {[(k, v) for k, v in top]}")
    print(f"    regime: {json.dumps(reg)}")
    results[name] = {"metrics": m, "pass": bool(p), "regime": reg}

json.dump(results, open("scripts/_miner3_cycle35b_results.json", "w"), indent=1, default=float)
print("\nDONE cycle35b")
