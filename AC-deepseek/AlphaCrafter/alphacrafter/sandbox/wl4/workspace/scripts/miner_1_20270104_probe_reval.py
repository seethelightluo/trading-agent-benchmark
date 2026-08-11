"""miner_1 2027-01-04: probe window + active library drift re-validation.

Current date 2027-01-04, data visible through 2027-01-01.
Library (4 effective): vol_price_corr_20 (+), dn_mkt_beta_60d (+),
eurusd_beta_60d (-), rate_beta_cn10y_60d (-).

Admission gate (shared 15-asset universe): |IC| >= 0.0070 AND |ICIR| >= 0.0840
at h=10, n_ic_dates >= 200, max_abs_library_correlation < 0.5.
No lookahead: factor at t uses data <= t; fwd = close[t+h]/close[t]-1.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, summarize_ic, coverage_metrics,
                                 turnover_rank, max_library_corr, decay_profile)

t0 = time.time()
panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
MIN_IC_DATES = 200
print(f"panels={len(panels)} closes={closes.shape} dates="
      f"{closes.index[0].date()}..{closes.index[-1].date()} load={time.time()-t0:.1f}s", flush=True)


def rank_ic_series_vec(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8):
    fr = factor_panel.rank(axis=1, method="average", na_option="keep")
    rr = fwd.rank(axis=1, method="average", na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1).astype(float)
    f = fr.where(valid)
    r = rr.where(valid)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = fm.mul(rm).sum(axis=1, min_count=1)
    den = np.sqrt(fm.pow(2).sum(axis=1, min_count=1) * rm.pow(2).sum(axis=1, min_count=1))
    ic = num.div(den.replace(0, np.nan))
    ic = ic.where((n >= min_valid) & (n >= 2) & (den > 1e-14))
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    ic.name = "ic"
    return ic


def summarize(ic_series, expected_sign=1):
    ic = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    icir = ic / std if std > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(std, 4)}


def evaluate(name, panel, exp_sign, lib, verbose=True):
    panel = panel.reindex(closes.index)
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_vec(panel, fwd, MIN_VALID)
    m = summarize(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, MIN_VALID, exp_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES and corr < 0.5)
    if verbose:
        print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
              f"to={m.get('turnover_10d_rank')} rho={corr:.3f}({key}) "
              f"decay10={m['decay_ic_by_horizon'].get('10')} {'-> PASS' if gate else ''}", flush=True)
    return m, ics


def per_asset(build, min_len=260):
    cols = {}
    for a in closes.columns:
        s = closes[a].dropna()
        if len(s) < min_len:
            continue
        f = build(s, a)
        if f is None or len(f) == 0:
            continue
        if isinstance(f, pd.Series):
            f = f.reindex(s.index)
        else:
            f = pd.Series(np.asarray(f), index=s.index[:len(f)])
        cols[a] = f
    return pd.DataFrame(cols).reindex(closes.index)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40, exclude_self=None):
    beta = {}
    for a in asset_ret.columns:
        if exclude_self is not None and a == exclude_self:
            beta[a] = pd.Series(np.nan, index=asset_ret.index)
            continue
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


# ---------------- current library signals (4 effective factors) ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
vix = panels["VIX"]["close"].astype(float)
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)


def vpc_build(s, a):
    r = s.pct_change()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    z = pd.concat([r.rename("a"), v.rename("v")], axis=1).dropna()
    return z["a"].rolling(20, min_periods=10).corr(z["v"])


lib = {
    "vol_price_corr_20": per_asset(vpc_build),
    "eurusd_beta_60d": rolling_beta(rets, eur.pct_change(), 60),
    "rate_beta_cn10y_60d": rolling_beta(rets, cn10.pct_change(), 60),
    "dn_mkt_beta_60d": rolling_beta(rets, dn, 60),
}
for k in lib:
    lib[k] = lib[k].reindex(closes.index)

# ============ 0. PROBE ============
print("\n=== 0. WINDOW ===", flush=True)
print(f"dates={closes.index[0].date()}..{closes.index[-1].date()} n_dates={len(closes)} "
      f"n_assets={closes.shape[1]}", flush=True)
for a in closes.columns:
    n = int(closes[a].notna().sum())
    print(f"  {a:10s} obs={n} last={closes[a].dropna().index[-1].date()}", flush=True)

# ============ 1. ACTIVE LIBRARY DRIFT ============
print("\n=== 1. ACTIVE LIBRARY RE-VALIDATION (drift, extended window) ===", flush=True)
active = {
    "vol_price_corr_20": (lib["vol_price_corr_20"], 1),
    "eurusd_beta_60d": (lib["eurusd_beta_60d"], -1),
    "rate_beta_cn10y_60d": (lib["rate_beta_cn10y_60d"], -1),
    "dn_mkt_beta_60d": (lib["dn_mkt_beta_60d"], 1),
}
active_meta = {}
for name, (panel, es) in active.items():
    m, ics = evaluate(f"[active]{name}", panel, es, lib)
    active_meta[name] = m

out = {"active": active_meta,
       "window": {"first": str(closes.index[0].date()), "last": str(closes.index[-1].date()),
                  "n_dates": int(len(closes)), "n_assets": int(closes.shape[1])}}
with open("scripts/_miner1_20270104_reval.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nsaved scripts/_miner1_20270104_reval.json elapsed={time.time()-t0:.1f}s")
