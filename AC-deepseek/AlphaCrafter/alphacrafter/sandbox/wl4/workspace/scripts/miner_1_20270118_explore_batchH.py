"""miner_1 2027-01-18: explore batch H (vectorized) - new decorrelated factor families.

Admission gate: |IC| >= 0.0070 AND |ICIR| >= 0.0840 at h=10,
n_ic_dates >= 200, max_abs_library_correlation < 0.5.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, coverage_metrics, turnover_rank,
                                 max_library_corr)

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


def rank_ic_series_vec(factor_panel, fwd, min_valid=MIN_VALID):
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


def summarize(ic_series):
    ic = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    return {"ic": round(ic, 4), "icir": round(ic / std, 4) if std > 0 else 0.0,
            "ic_hit_ratio": round(float((np.sign(ic_series) != 0).mean()), 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(std, 4)}


def per_asset(build, min_len=260):
    cols = {}
    for a in closes.columns:
        s = closes[a].dropna()
        if len(s) < min_len:
            continue
        f = build(s, a)
        if f is None or len(f) == 0:
            continue
        f = f.reindex(s.index) if isinstance(f, pd.Series) else pd.Series(np.asarray(f), index=s.index[:len(f)])
        cols[a] = f
    return pd.DataFrame(cols).reindex(closes.index)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


# ---------------- library signals (4 effective) ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
dxy = panels["DXY"]["close"].astype(float)
jpy = panels["USDJPY"]["close"].astype(float)


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

# ---------------- candidate builders (vectorized) ----------------
def er_build(s, a, win=20):
    r = s.pct_change()
    num = (s - s.shift(win)).abs()
    den = r.abs().rolling(win).sum()
    return (num / den).where(den > 1e-12)

def autocorr_build(s, a, win=60):
    r = s.pct_change()
    r1 = r.shift(1)
    cov = r.rolling(win, min_periods=40).cov(r1)
    var = r.rolling(win, min_periods=40).var()
    return (cov / var).where(var > 1e-14)

def skew_build(s, a, win=60):
    return s.pct_change().rolling(win, min_periods=40).skew()

def down_ratio_build(s, a, win=60):
    r = s.pct_change()
    neg = r.where(r < 0, 0.0)
    down = neg.rolling(win, min_periods=40).std()
    tot = r.rolling(win, min_periods=40).std()
    return (down / tot).where(tot > 1e-14)

def upday_build(s, a, win=60):
    return (s.pct_change() > 0).astype(float).rolling(win, min_periods=40).mean()

def amplitude_build(s, a, win=20):
    h = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    return ((h - lo) / s).rolling(win, min_periods=10).mean()

def range_pos_build(s, a, win=20):
    h = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    hh = h.rolling(win, min_periods=10).max()
    ll = lo.rolling(win, min_periods=10).min()
    return ((s - ll) / (hh - ll)).where((hh - ll) > 1e-12)

def rel_mom_build(s, a, win=20):
    mom = s / s.shift(win) - 1.0
    cm = mom.rolling(252, min_periods=60).mean()
    return mom - cm

def sharpe_build(s, a, win=60):
    r = s.pct_change()
    mu = r.rolling(win, min_periods=40).mean()
    sd = r.rolling(win, min_periods=40).std()
    return (mu / sd).where(sd > 1e-14)

def max_dd_build(s, a, win=60):
    roll_max = s.rolling(win, min_periods=40).max()
    dd = s / roll_max - 1.0
    return dd.rolling(win, min_periods=40).min()

candidates = {
    "er_20": per_asset(er_build),
    "autocorr_1_60": per_asset(autocorr_build),
    "skew_60": per_asset(skew_build),
    "down_ratio_60": per_asset(down_ratio_build),
    "upday_ratio_60": per_asset(upday_build),
    "amplitude_20": per_asset(amplitude_build),
    "dxy_beta_60": rolling_beta(rets, dxy.pct_change(), 60),
    "usdjpy_beta_60": rolling_beta(rets, jpy.pct_change(), 60),
    "range_pos_20": per_asset(range_pos_build),
    "rel_mom_20": per_asset(rel_mom_build),
    "sharpe_60": per_asset(sharpe_build),
    "max_dd_60": per_asset(max_dd_build),
}
for k in candidates:
    candidates[k] = candidates[k].reindex(closes.index)

# ---------------- evaluate (single IC pass per candidate, report both signs) ----------------
print("\n=== batch H candidates (h=10 admission) ===", flush=True)
fwd = forward_returns(closes, H_ADM)
results = {}
for name, panel in candidates.items():
    ics = rank_ic_series_vec(panel, fwd)
    m = summarize(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    dec = {}
    for h in HORIZONS:
        ic_h = rank_ic_series_vec(panel, forward_returns(closes, h))
        dec[str(h)] = round(float(ic_h.mean()), 4) if len(ic_h) else None
    m["decay_ic_by_horizon"] = dec
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    m["expected_sign"] = 1 if m["ic"] >= 0 else -1
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES and corr < 0.5)
    print(f"{name:16s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m.get('turnover_10d_rank')} rho={corr:.3f}({key}) "
          f"decay10={dec.get('10')} {'-> PASS' if gate else ''}", flush=True)
    results[name] = m

with open("scripts/_miner1_20270118_batchH.json", "w") as fh:
    json.dump({"window": {"first": str(closes.index[0].date()), "last": str(closes.index[-1].date()),
                          "n_dates": int(len(closes)), "n_assets": int(closes.shape[1])},
               "results": results}, fh, indent=1, default=str)
print(f"\nsaved scripts/_miner1_20270118_batchH.json elapsed={time.time()-t0:.1f}s")
