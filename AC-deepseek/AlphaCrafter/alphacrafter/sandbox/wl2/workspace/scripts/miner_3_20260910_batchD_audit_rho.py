"""miner_3 2026-09-10 batch D follow-up: proper AUDIT-STYLE correlation check.

The deterministic post-Miner gate recomputes pairwise rho from real signal
artifacts using MEAN DAILY cross-sectional Spearman (see
miner_1_20260813_active_lib_audit.py: mean_daily_rho). The quick library_pairwise_corr
snapshot (first valid row) OVERSTATES correlation. Recompute candidates with
time-averaged rho vs ALL 30 factors/*.signal.npy artifacts and vs KEPT members.

Also fix beta_asym_60 (down-day mask was never applied -> identically zero).
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, fwd_by_horizon_dict, MIN_ASSETS)

GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
print(f"grid rows={len(GRID)} last={GRID[-1]} assets={len(series)}/15")

# --- kept library ids: JSON present in factors/ with EFFECTIVE status (not evicted) ---
KEPT = set()
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f or "ensemble" in f:
        continue
    try:
        d = json.load(open(f))
        if d.get("validation", {}).get("status") == "EFFECTIVE":
            KEPT.add(d["factor_id"])
    except Exception:
        pass
print("kept ids:", len(KEPT), sorted(KEPT))

ARTIFACTS = sorted(glob.glob("factors/*.signal.npy"))


def mean_daily_rho(a, b, min_assets=8):
    rows = min(a.shape[0], b.shape[0])
    rhos = []
    for t in range(rows):
        x, y = a[t], b[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < min_assets:
            continue
        xs = pd.Series(x[ok]).rank()
        ys = pd.Series(y[ok]).rank()
        c = xs.corr(ys)
        if np.isfinite(c):
            rhos.append(c)
    return float(np.mean(rhos)) if rhos else 0.0


def rho_table(mat):
    out = {}
    for f in ARTIFACTS:
        fid = os.path.basename(f).replace(".signal.npy", "")
        arr = np.load(f, allow_pickle=True)
        out[fid] = mean_daily_rho(mat, arr)
    mx = max(out.items(), key=lambda kv: abs(kv[1]))
    kept_max = max((abs(v) for k, v in out.items() if k in KEPT), default=0.0)
    return out, mx, kept_max


def rolling_beta_reg(a, b, w=60, minp=40):
    joined = pd.concat([a, b], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(w, min_periods=minp).cov(joined["b"])
    var = joined["b"].rolling(w, min_periods=minp).var()
    beta = pd.Series(safe_div(cov, var), index=joined.index)
    ma = joined["a"].rolling(w, min_periods=minp).mean()
    mb = joined["b"].rolling(w, min_periods=minp).mean()
    resid = joined["a"] - (ma - beta * mb + beta * joined["b"])
    out = pd.DataFrame({"beta": beta, "resid": resid,
                        "va": joined["a"].rolling(w, min_periods=minp).var(),
                        "vb": var}, index=joined.index)
    return out.reindex(a.index)


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)


spx_ret = series["SPX"]["ret"]

# build candidates
cands = {}

# 1. idio_mom_20x60
d = {}
for s, df in series.items():
    reg = rolling_beta_reg(df["ret"], spx_ret)
    d[s] = pd.Series(reg["resid"].rolling(20, min_periods=10).sum().shift(5), index=df.index)
cands["idio_mom_20x60"] = to_grid(d)

# 2. r2_spx_60
d = {}
for s, df in series.items():
    reg = rolling_beta_reg(df["ret"], spx_ret)
    r2 = (reg["beta"] ** 2) * reg["vb"] / reg["va"]
    d[s] = pd.Series(np.clip(1.0 - r2, 0.0, 1.0), index=df.index)
cands["r2_spx_60"] = to_grid(d)

# 3. adx_14
d = {}
for s, df in series.items():
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    s_tr = tr.rolling(14, min_periods=14).mean()
    s_pdm = pd.Series(pdm, index=df.index).rolling(14, min_periods=14).mean()
    s_ndm = pd.Series(ndm, index=df.index).rolling(14, min_periods=14).mean()
    pdi = 100 * s_pdm / s_tr; ndi = 100 * s_ndm / s_tr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    d[s] = dx.rolling(14, min_periods=14).mean()
cands["adx_14"] = to_grid(d)

# 4. aroon_25
d = {}
for s, df in series.items():
    up = df["high"].rolling(25, min_periods=15).apply(
        lambda x: 100.0 * (len(x) - 1 - np.argmax(x)) / (len(x) - 1), raw=True)
    dn = df["low"].rolling(25, min_periods=15).apply(
        lambda x: 100.0 * (len(x) - 1 - np.argmin(x)) / (len(x) - 1), raw=True)
    d[s] = (up - dn).astype(float)
cands["aroon_25"] = to_grid(d)

# 5. beta_up_spx_60 (up-day beta) and 6. beta_asym_60 (down - up, FIXED)
def cond_beta(df, mkt_ret, up=True, w=60, minp=30):
    joined = pd.concat([df["ret"], mkt_ret], axis=1, join="outer")
    joined.columns = ["a", "b"]
    mask = joined["b"] > 0 if up else joined["b"] < 0
    a_c = joined["a"].where(mask)
    b_c = joined["b"].where(mask)
    cov = a_c.rolling(w, min_periods=minp).cov(b_c)
    var = b_c.rolling(w, min_periods=minp).var()
    return pd.Series(safe_div(cov, var), index=joined.index).reindex(df.index)


d = {}
for s, df in series.items():
    d[s] = cond_beta(df, spx_ret, up=True)
cands["beta_up_spx_60"] = to_grid(d)

d = {}
for s, df in series.items():
    db = cond_beta(df, spx_ret, up=False)
    ub = cond_beta(df, spx_ret, up=True)
    d[s] = pd.Series(db - ub, index=df.index)
cands["beta_asym_60"] = to_grid(d)

# validate each
out = {}
for name, mat in cands.items():
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO IC DATES")
        continue
    ic, icir = summ["ic"], summ["icir"]
    ok_gate = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    rho_tbl, mx, kept_max = rho_table(mat)
    top = sorted(rho_tbl.items(), key=lambda kv: -abs(kv[1]))[:5]
    ok_corr = kept_max < 0.5
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"GATE={ok_gate} | kept_max_rho={kept_max:.3f} all_max_rho={abs(mx[1]):.3f} ({mx[0]}) ADMIT={ok_gate and ok_corr}")
    print("   top5 rho:", [(k, round(v, 3)) for k, v in top])
    out[name] = {"ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
                 "n_ic_dates": summ["n_ic_dates"], "pass_gate_icir": bool(ok_gate),
                 "kept_max_rho": round(kept_max, 4), "all_max_rho": round(abs(mx[1]), 4),
                 "all_max_name": mx[0], "admit": bool(ok_gate and ok_corr),
                 "rho_all": {k: round(v, 4) for k, v in rho_tbl.items()}}

json.dump(out, open("scripts/miner_3_20260910_batchD_audit_rho.json", "w"), indent=1)
print("SAVED")
