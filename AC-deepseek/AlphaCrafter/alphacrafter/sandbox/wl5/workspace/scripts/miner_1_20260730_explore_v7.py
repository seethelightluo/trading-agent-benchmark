"""miner_1 cycle 2026-07-30: re-validate strongest known factors + new candidates.
Persistence-ready: saves full signal matrices (dates x assets) as artifacts.
All research restricted to visible window <= 2026-07-29.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns, ic_series,
                             summary_metrics, library_ic_series_map, max_abs_library_corr,
                             regime_split)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
lp = np.log(close)

def rolling_trend_components(df, win):
    """Rolling OLS of log-close on time -> n, r2, slope, sign using sums."""
    t = np.arange(len(df))
    st = pd.Series(t, index=df.index)
    s = df.rolling(win, min_periods=1).sum()
    n = df.rolling(win, min_periods=1).count()
    with np.errstate(all="ignore"):
        mean_y = s / n
        st_w = st.rolling(win, min_periods=1).sum() / n
        sty = (df.mul(st, axis=0)).rolling(win, min_periods=1).sum() / n
        sy2 = (df ** 2).rolling(win, min_periods=1).sum() / n
        st2 = (st ** 2).rolling(win, min_periods=1).sum() / n
        cov = sty - mean_y * st_w
        var_t = st2 - st_w ** 2
        var_y = sy2 - mean_y ** 2
        r2 = (cov ** 2) / (var_t * var_y)
        slope_sign = np.sign(cov)
    mp = int(win * 0.6)
    n = n.where(n >= mp)
    r2 = r2.where((var_t > 1e-12) & (var_y > 1e-12))
    return n, r2, slope_sign


def tstat(n, r2, slope_sign):
    ts = np.sqrt((n - 2) * r2 / (1.0 - r2)) * slope_sign
    return ts.where(r2 < 0.999)


# ---- trend components ----
n30, r2_30, sg30 = rolling_trend_components(lp, 30)
t30 = tstat(n30, r2_30, sg30)
n60, r2_60, sg60 = rolling_trend_components(lp, 60)
t60 = tstat(n60, r2_60, sg60)

# ---- momentum ----
mom10 = lp.diff(10)
mom20 = lp.diff(20)
mom60 = lp.diff(60)
mom120 = lp.diff(120)

# ---- vols ----
vol20 = ret.rolling(20, min_periods=12).std() * np.sqrt(252)
vol60 = ret.rolling(60, min_periods=36).std() * np.sqrt(252)
down = ret.where(ret < 0, 0.0)
downside_dev60 = np.sqrt((down ** 2).rolling(60, min_periods=36).mean()) * np.sqrt(252)

# ---- macro ----
macro = macro_closes(VIS)
vix = macro["VIX"]
vix_ret = vix.pct_change()
dxy_ret = macro["DXY"].pct_change()
us10 = close["US10Y"]
cn10 = close["CN10Y"]

# ---- rolling beta helpers ----
def rolling_beta(asset_ret, mkt_ret, win, mp=36):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=mp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=mp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)

def rolling_corr(asset_ret, mkt_ret, win, mp=36):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        c = pair["a"].rolling(win, min_periods=mp).corr(pair["m"])
        out[a] = c
    return pd.DataFrame(out).reindex(asset_ret.index)


btc_ret = ret["BTC"]
beta_btc60 = rolling_beta(ret, btc_ret, 60)
beta_vix60 = rolling_beta(ret, vix_ret, 60)
corr_us10_60 = rolling_corr(ret, us10.pct_change(), 60)
corr_dxy_60 = rolling_corr(ret, dxy_ret, 60)

# ---- new candidates ----
sma60 = close.rolling(60, min_periods=36).mean()
sma120 = close.rolling(120, min_periods=72).mean()
roll_max60 = close.rolling(60, min_periods=36).max()
roll_min20 = close.rolling(20, min_periods=12).min()
roll_max20 = close.rolling(20, min_periods=12).max()

cands = {
    # known strong re-validation
    "trend_tstat_30": t30,
    "trend_tstat_60": t60,
    "risk_on_alpha_20x60": None,  # built below
    "mom_10d_skip5": close.shift(5) / close.shift(15) - 1.0,
    "mom_120d_skip5": close.shift(5) / close.shift(125) - 1.0,
    "vol_of_vol20x60": ret.rolling(20, min_periods=12).std().rolling(60, min_periods=36).std(),
    "vix_beta_cond_60x20": None,  # built below
    # new: carry / deviation
    "carry_60d": close / sma60 - 1.0,
    "carry_120d": close / sma120 - 1.0,
    "drawdown_60d": close / roll_max60 - 1.0,
    "stoch_pos_20d": (close - roll_min20) / (roll_max20 - roll_min20),
    # new: risk-adjusted
    "sharpe_63d": ret.rolling(63, min_periods=38).mean() / ret.rolling(63, min_periods=38).std(),
    "downside_share_60": downside_dev60 / vol60,
    "mom20_vol20": mom20 / vol20,
    # new: cross-asset sensitivity
    "btc_beta_60": beta_btc60,
    "vix_beta_60": beta_vix60,
    "us10y_corr_60": corr_us10_60,
    "dxy_corr_60": corr_dxy_60,
}

# risk_on_alpha: 20d return minus beta*(risk-on basket 20d return)
riskon = lp[["SPX", "NDX", "SOX", "N225", "HSI", "SX5E", "000300.SH", "000688.SH"]].mean(axis=1)
riskon_ret20 = riskon.diff(20)
beta_riskon = rolling_beta(ret, riskon.diff(), 60)
cands["risk_on_alpha_20x60"] = mom20 - beta_riskon.mul(riskon_ret20, axis=0)

# vix_beta_cond: -beta_vix60 * vix 20d change (risk-on when VIX falling and asset low-vix-beta)
cands["vix_beta_cond_60x20"] = -beta_vix60.mul(vix / vix.shift(20) - 1.0, axis=0)

fr = forward_returns(close, H)
lib = library_ic_series_map(close, h=H)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS} lib_size={len(lib)}\n")

results = {}
signal_store = {}
for fid, sig in cands.items():
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{fid}: INSUFFICIENT ({len(ic_s)} ic dates; cells={int(sig.notna().sum().sum())})")
        results[fid] = {"gate_pass": False, "reason": "insufficient IC dates",
                        "n_ic_dates": len(ic_s), "valid_entries": int(sig.notna().sum().sum())}
        continue
    m["max_abs_library_correlation"] = max_abs_library_corr(ic_s, lib)
    m["regime"] = regime_split(ic_s)
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    m["gate_pass"] = gate
    results[fid] = m
    # persist signal artifact (long-form records, compact)
    long = sig.stack().dropna().reset_index()
    long.columns = ["date", "symbol", "value"]
    long["date"] = long["date"].astype(str)
    long["value"] = long["value"].round(6)
    signal_store[fid] = long.to_dict(orient="records")
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} max_rho_lib={m['max_abs_library_correlation']} GATE={gate} artifacts={len(signal_store[fid])}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

with open("scripts/miner_1_20260730_explore_v7_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
with open("scripts/miner_1_20260730_explore_v7_signals.json", "w") as f:
    json.dump(signal_store, f)
print("\nDONE saved results + signals")
