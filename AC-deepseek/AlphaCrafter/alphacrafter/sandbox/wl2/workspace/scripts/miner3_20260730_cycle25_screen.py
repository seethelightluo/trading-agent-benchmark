"""miner_3 cycle25 screen: novel low-correlation factor candidates.

Focus: intraday/overnight decomposition, candle-position sentiment, tail
asymmetry, liquidity (Amihud), short-term vol expansion, FX-conditional
betas (USDCNY stress, EURUSD risk-on), SPX downside beta, gap frequency.

All validated with the shared miner3 framework (visible-through 2026-07-29,
15-asset cross-section, 10d forward horizon, >=8 valid assets per date).
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, max_abs_library_correlation,
                        ADMISSION_HORIZON, VISIBLE, MIN_ASSETS)

panel = pd.DataFrame(build_panel())
ret = panel.pct_change()
EPS = 1e-9


def macro_beta_cond(asset, beta_win=60, macro_win=20):
    obs = pd.read_csv(f'../persistent/index_data/{asset}.csv', parse_dates=['date'])
    obs = obs[obs['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    obs_r = obs.pct_change()
    beta = ret.rolling(beta_win, min_periods=30).cov(obs_r) / obs_r.rolling(beta_win, min_periods=30).var()
    mom = obs / obs.shift(macro_win) - 1.0
    return beta.mul(mom, axis=0).reindex(panel.index)


def downside_beta_spx(win=60):
    spx = panel['SPX'].pct_change()
    d = spx < 0
    # per-asset beta computed only on SPX-down days via cov/var trick
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for a in panel.columns:
        a_r = ret[a]
        cov = a_r.where(d).rolling(win, min_periods=20).cov(spx.where(d))
        var = spx.where(d).rolling(win, min_periods=20).var()
        out[a] = cov / (var + EPS)
    return out


candidates = {
    'intraday_str_20': lambda df: ((df / df['open'].reindex(df.index)) - 1.0).rolling(20, min_periods=10).mean(),
    'overnight_gap_20': lambda df: ((df['open'].reindex(df.index) / df.shift(1)) - 1.0).rolling(20, min_periods=10).mean(),
    'candle_pos_20': lambda df: ((df - df['low'].reindex(df.index)) / (df['high'].reindex(df.index) - df['low'].reindex(df.index) + EPS)).rolling(20, min_periods=10).mean(),
    'body_ratio_20': lambda df: ((df - df['open'].reindex(df.index)).abs() / (df['high'].reindex(df.index) - df['low'].reindex(df.index) + EPS)).rolling(20, min_periods=10).mean(),
    'tail_ratio_60': lambda df: ret.rolling(60, min_periods=30).quantile(0.95) / (ret.rolling(60, min_periods=30).quantile(0.05).abs() + EPS),
    'amihud_illiq_20': lambda df: (ret.abs() / (panel['volume'].reindex(panel.index) + EPS)).rolling(20, min_periods=10).mean(),
    'vol_ratio_5_60': lambda df: ret.rolling(5, min_periods=3).std() / (ret.rolling(60, min_periods=30).std() + EPS),
    'gap_freq_20': lambda df: (df['open'].reindex(df.index) / df.shift(1) - 1.0).abs().gt(0.01).rolling(20, min_periods=10).mean().astype(float),
    'usdcny_beta_cond_60x20': lambda df: macro_beta_cond('USDCNY'),
    'eurusd_beta_cond_60x20': lambda df: macro_beta_cond('EURUSD'),
    'downside_beta_spx_60': lambda df: downside_beta_spx(60),
    'range_amp_z_20': lambda df: ((df['high'].reindex(df.index) - df['low'].reindex(df.index)) / df).rolling(20, min_periods=10).mean(),
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
    valid = factor_df.notna().sum().sum()
    cov = valid / (factor_df.shape[0] * factor_df.shape[1])
    n_ge8 = sum(1 for d in factor_df.index if factor_df.loc[d].notna().sum() >= MIN_ASSETS)
    cov_ge8 = n_ge8 / len(factor_df)
    turn = mean_rank_turnover(factor_df)
    maxrho, rho_names = max_abs_library_correlation(factor_df)
    gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084) and (maxrho < 0.5)
    results[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                        cov_ge8=cov_ge8, turn=turn, maxrho=maxrho, rho_names=rho_names, gate=gate)
    print(f'{fid:24s} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n={len(ic_series):4d} '
          f'cov={cov:.3f} ge8={cov_ge8:.3f} turn={turn:.3f} maxrho={maxrho:.3f} rho={rho_names} gate={gate}')

with open('scripts/miner3_cycle25_screen_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print('\nsaved scripts/miner3_cycle25_screen_results.json')
