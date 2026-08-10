"""miner_3 cycle 24 screen v2: new orthogonal factor families (NaN-robust).

Fixes from v1:
- rolling apply with corrcoef broken on mixed calendars -> use min_periods /
  shift-based NaN-robust definitions (same convention as prior cycles).
- macro-conditional candidates called as lambdas (dict evaluated them eagerly).

Families not covered by library: trend efficiency, drawdown depth, return
autocorrelation (lag-1 product proxy), skewness, lottery/upside risk,
USDJPY macro-conditional beta, VIX-conditioned momentum, Bollinger bandwidth.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, max_abs_library_correlation,
                        ADMISSION_HORIZON, HORIZONS, VISIBLE)

panel = build_panel()
RET = 'pct_change'
ret = panel.pct_change()

def kaufman_eff(win):
    def f(df):
        num = (df - df.shift(win)).abs()
        den = df.pct_change().abs().rolling(win, min_periods=max(10, win // 3)).sum()
        return num / (den + 1e-9)
    return f

def dd_depth(win):
    def f(df):
        return df / df.rolling(win, min_periods=max(20, win // 4)).max() - 1.0
    return f

def skew_60(df):
    return ret.rolling(60, min_periods=30).skew()

def acorr_lag1_proxy(win):
    def f(df):
        return (ret * ret.shift(1)).rolling(win, min_periods=max(5, win // 2)).mean()
    return f

def lottery_max_ret(win):
    def f(df):
        return -ret.rolling(win, min_periods=win // 2).max()
    return f

def boll_bw(win):
    def f(df):
        return df.rolling(win, min_periods=win // 2).std() / (df.rolling(win, min_periods=win // 2).mean() + 1e-9)
    return f

def rel_strength(win):
    def f(df):
        r = df.pct_change(win)
        return r.sub(r.median(axis=1), axis=0)
    return f

def macro_beta_cond(asset, beta_win=60, macro_win=20):
    obs = pd.read_csv(f'../persistent/index_data/{asset}.csv', parse_dates=['date'])
    obs = obs[obs['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    obs_r = obs.pct_change()
    beta = ret.rolling(beta_win, min_periods=30).cov(obs_r) / obs_r.rolling(beta_win, min_periods=30).var()
    mom = obs / obs.shift(macro_win) - 1.0
    out = beta.mul(mom, axis=0)
    return out.reindex(panel.index)

def vix_cond_mom(short_win=20):
    vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date'])
    vix = vix[vix['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    vix_z = (vix - vix.rolling(60, min_periods=20).mean()) / (vix.rolling(60, min_periods=20).std() + 1e-9)
    damp = 1.0 / (1.0 + vix_z.clip(lower=0) ** 2)
    mom = panel.pct_change(short_win)
    return mom.mul(damp, axis=0)

candidates = {
    'kaufman_eff_60':  kaufman_eff(60),
    'kaufman_eff_120': kaufman_eff(120),
    'dd_depth_60':     dd_depth(60),
    'dd_depth_252':    dd_depth(252),
    'skew_60':         skew_60,
    'acorr_lag1_10':   acorr_lag1_proxy(10),
    'lottery_max_ret_20': lottery_max_ret(20),
    'boll_bw_20':      boll_bw(20),
    'rel_strength_60': rel_strength(60),
    'usdjpy_beta_cond_60x20': lambda df: macro_beta_cond('USDJPY'),
    'vix_cond_mom_20': lambda df: vix_cond_mom(20),
}

fwd10 = forward_returns(panel, ADMISSION_HORIZON)
results = {}
for fid, fn in candidates.items():
    try:
        factor_df = fn(panel)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}')
        continue
    ic_series = spearman_ic(factor_df, fwd10)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    cov = float(factor_df.notna().sum().sum() / (factor_df.shape[0] * factor_df.shape[1]))
    n_ge8 = sum(1 for d in factor_df.index if factor_df.loc[d].notna().sum() >= 8)
    turn = mean_rank_turnover(factor_df)
    maxrho, rho_names = max_abs_library_correlation(factor_df)
    decay = {}
    for h in HORIZONS:
        fh = forward_returns(panel, h)
        s = spearman_ic(factor_df, fh)
        decay[str(h)] = round(float(s.mean()), 4)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                        n_dates_ge8=n_ge8, turn=turn, maxrho=maxrho,
                        rho_names=rho_names, decay=decay, gate=bool(gate))
    print(f'[{"PASS" if gate else "fail"}] {fid}: ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} '
          f'n={len(ic_series)} cov={cov:.3f} dates_ge8={n_ge8} turn={turn:.3f} '
          f'maxrho={maxrho:.3f} vs {rho_names} decay10={decay["10"]}')

with open('scripts/miner3_cycle24_results.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=float)
print('\nSaved scripts/miner3_cycle24_results.json')
