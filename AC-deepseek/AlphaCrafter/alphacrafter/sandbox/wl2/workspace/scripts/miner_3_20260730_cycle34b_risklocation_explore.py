"""miner_3 cycle 34b: explore orthogonal risk-location axes (post-persistence).

New candidates (avoid momentum/trend family - crowded after gain_loss/hilo/rsi evictions):
  1. overnight_vol_ratio_60 : std(overnight gap) / std(intraday return) - where does risk live?
  2. down_vol_ratio_60      : downside semi-dev / total vol - normalized tail tilt
  3. parkinson_ratio_60     : parkinson vol (OHLC range) / close-to-close vol - range efficiency
  4. gap_freq_20            : fraction of days |overnight gap| > 1.5*20d vol - gap intensity
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, forward_returns, compute_ic,
                         validate_factor, VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

ACTIVE = ["calmness_20", "dxy_beta_cond_60x20", "intraday_drift_20",
          "mom20_volproxy60", "usdjpy_beta_cond_120x60"]
lib = {}
for fid in ACTIVE:
    a = np.load(Path("factors") / f"{fid}.signal.npy")
    lib[fid] = pd.DataFrame(a, index=panel.index, columns=panel.columns)


def stacked_spearman(a, b):
    df = pd.concat([a.stack().rename("x"), b.stack().rename("y")], axis=1).dropna()
    return float(df["x"].corr(df["y"], method="spearman")) if len(df) >= 30 else 0.0


def load_ohlc():
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date").set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


OHLC = load_ohlc()


def build(func):
    out = {}
    for a in TRADABLES:
        out[a] = func(OHLC[a].dropna()).reindex(panel.index)
    F = pd.DataFrame(out, index=panel.index)
    return F


def ovn_vol_ratio(df, w=60, mp=40):
    gap = df["open"] / df["close"].shift(1) - 1.0
    intra = df["close"] / df["open"] - 1.0
    sg = gap.rolling(w, min_periods=mp).std()
    si = intra.rolling(w, min_periods=mp).std()
    return sg / (si + 1e-12)


def down_vol_ratio(df, w=60, mp=40):
    r = df["close"].pct_change()
    dn = r.where(r < 0, 0.0).pow(2).rolling(w, min_periods=mp).mean().pow(0.5)
    tot = r.rolling(w, min_periods=mp).std()
    return dn / (tot + 1e-12)


def parkinson_ratio(df, w=60, mp=40):
    c = df["close"]
    hi = df["high"].rolling(w, min_periods=mp).max()
    lo = df["low"].rolling(w, min_periods=mp).min()
    rng = (hi - lo) / c
    r = c.pct_change()
    cc = r.rolling(w, min_periods=mp).std()
    return rng / (cc + 1e-12)


def gap_freq(df, w=20, mp=10, mult=1.5):
    gap = df["open"] / df["close"].shift(1) - 1.0
    vol = df["close"].pct_change().rolling(20, min_periods=10).std()
    flag = (gap.abs() > mult * vol).astype(float)
    return flag.rolling(w, min_periods=mp).mean()


cands = {
    "overnight_vol_ratio_60": build(ovn_vol_ratio),
    "down_vol_ratio_60": build(down_vol_ratio),
    "parkinson_ratio_60": build(parkinson_ratio),
    "gap_freq_20": build(gap_freq),
}

print("\n=== VALIDATION (admission horizon=10d, gate-style stacked rho) ===")
results = {}
for name, F in cands.items():
    m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    lc = {k: round(stacked_spearman(F, sig), 4) for k, sig in lib.items()}
    m["max_abs_library_correlation"] = round(max((abs(v) for v in lc.values()), default=0.0), 4)
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
    p_corr = abs(m["max_abs_library_correlation"]) < 0.5
    p = p_ic and p_corr
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxrho={m['max_abs_library_correlation']} "
          f"=> {'PASS' if p else 'FAIL'}")
    print(f"    pairwise: {m['library_pairwise_corr']}")
    print(f"    regime: {json.dumps(reg)}")
    results[name] = {"metrics": m, "pass": bool(p), "regime": reg}

json.dump(results, open("scripts/_miner3_cycle34b_results.json", "w"), indent=1, default=float)
print("\nDONE cycle34b")
