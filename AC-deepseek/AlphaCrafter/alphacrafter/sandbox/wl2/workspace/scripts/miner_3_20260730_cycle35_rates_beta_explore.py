"""miner_3 cycle 35: explore RATES-beta and microstructure axes orthogonal to the current library.

Active library (EFFECTIVE with .signal.npy): calmness_20, dxy_beta_cond_60x20,
usdjpy_beta_cond_120x60, mom20_volproxy60, downbeta_spx_60, lagbeta_spx_60,
vol_price_corr_60, intraday_drift_20, days_since_high_60, max_consec_gain_20,
max_consec_loss_20, gain_loss_20.

Candidates (new axes):
  A. rates_beta_cond_60x20   : beta(asset, US10Y_ret, 60) x US10Y 20d momentum
  B. cn10y_beta_cond_60x20   : beta(asset, CN10Y_ret, 60) x CN10Y 20d momentum
  C. ew_beta_60              : beta to equal-weight universe return (market beta)
  D. mdd_60                  : 60d max drawdown depth (not speed)
  E. gap_cont_ratio_60       : fraction of days where overnight gap and intraday move agree
  F. ovn_intra_corr_60       : 60d corr(overnight gap, intraday return)
  G. skew_60_fixed           : rolling 60d skewness (fixed coverage, per-asset calendar)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor,
                         VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------- active library: ALL effective jsons with signal artifacts ----------------
lib = {}
for jp in sorted(Path("factors").glob("*.json")):
    if jp.name.endswith(".bak") or jp.name == "factor_ensemble.json":
        continue
    try:
        d = json.load(open(jp))
    except Exception:
        continue
    if d.get("validation", {}).get("status") not in ("EFFECTIVE", "ACTIVE"):
        continue
    art = d.get("signal_artifact")
    if not art:
        continue
    p = Path("factors") / art
    if not p.exists():
        continue
    a = np.load(p)
    if a.shape != (len(panel), len(TRADABLES)):
        print(f"[lib skip] {d['factor_id']} shape {a.shape}")
        continue
    lib[d["factor_id"]] = pd.DataFrame(a, index=panel.index, columns=panel.columns)
print(f"active library artifacts ({len(lib)}): {list(lib.keys())}")


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


def beta_to(s, m_ret, w=60, minp=30):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), m_ret.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b.reindex(panel.index)


def cond_beta_factor(macro_name, w=60, mw=20):
    m = macro_series(macro_name)
    m_ret = m.pct_change()
    m_mom = m / m.shift(mw) - 1.0
    parts = {}
    for a in TRADABLES:
        parts[a] = beta_to(panel[a].dropna(), m_ret, w).mul(m_mom.reindex(panel.index), axis=0)
    return pd.DataFrame(parts, index=panel.index)


def ew_beta_60(s, ew_ret, w=60, minp=30):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), ew_ret.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b.reindex(panel.index)


def mdd_60(s, w=60, minp=40):
    r = s.pct_change()
    xa = (1 + r.fillna(0.0)).cumprod()
    peak = xa.rolling(w, min_periods=minp).max()
    return (xa / peak - 1.0).reindex(panel.index)


def build_ohlc(func):
    out = {}
    for a in TRADABLES:
        out[a] = func(OHLC[a].dropna()).reindex(panel.index)
    F = pd.DataFrame(out, index=panel.index)
    return F


def gap_cont_ratio(df, w=60, mp=40):
    gap = df["open"] / df["close"].shift(1) - 1.0
    intra = df["close"] / df["open"] - 1.0
    same = (np.sign(gap) == np.sign(intra)).astype(float)
    return same.rolling(w, min_periods=mp).mean()


def ovn_intra_corr(df, w=60, mp=40):
    gap = df["open"] / df["close"].shift(1) - 1.0
    intra = df["close"] / df["open"] - 1.0
    return gap.rolling(w, min_periods=mp).corr(intra)


def skew_60(s, w=60, mp=40):
    return s.pct_change().rolling(w, min_periods=mp).skew()


cands = {}
cands["rates_beta_cond_60x20"] = cond_beta_factor("US10Y") if False else None
# US10Y is a tradable itself; use macro_series if available else asset column
try:
    us10y_ser = macro_series("US10Y")
    if us10y_ser is None:
        raise ValueError
except Exception:
    us10y_ser = panel["US10Y"]
cands["rates_beta_cond_60x20"] = cond_beta_factor("US10Y") if "US10Y" in [x for x in dir()] else None
cands["rates_beta_cond_60x20"] = None  # placeholder, built below

# rebuild rates beta with explicit series (US10Y is both tradable and macro)
def cond_beta_from_series(m_ser, w=60, mw=20):
    m_ret = m_ser.pct_change()
    m_mom = m_ser / m_ser.shift(mw) - 1.0
    parts = {}
    for a in TRADABLES:
        b = beta_to(panel[a].dropna(), m_ret, w)
        parts[a] = b.mul(m_mom.reindex(b.index), axis=0).reindex(panel.index)
    return pd.DataFrame(parts, index=panel.index)


cands["rates_beta_cond_60x20"] = cond_beta_from_series(panel["US10Y"].dropna())
cands["cn10y_beta_cond_60x20"] = cond_beta_from_series(panel["CN10Y"].dropna())

# EW universe return: mean pct change across the 15 tradables
rets = panel.pct_change()
ew_ret = rets.mean(axis=1, skipna=True)
cands["ew_beta_60"] = per_asset(panel, ew_beta_60, ew_ret)

cands["mdd_60"] = per_asset(panel, mdd_60)
cands["gap_cont_ratio_60"] = build_ohlc(gap_cont_ratio)
cands["ovn_intra_corr_60"] = build_ohlc(ovn_intra_corr)
cands["skew_60_fixed"] = per_asset(panel, skew_60)

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

json.dump(results, open("scripts/_miner3_cycle35_results.json", "w"), indent=1, default=float)
print("\nDONE cycle35")
