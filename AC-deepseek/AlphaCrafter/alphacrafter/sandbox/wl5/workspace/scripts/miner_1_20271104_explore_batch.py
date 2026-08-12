"""miner_1 2027-11-04: explore new factor batch.

Ideas (orthogonal to existing library which covers trend quality, semi-deviation,
plain momentum 10/120d, time-under-water, vol-of-vol, DXY/WTI/VIX betas, kurtosis,
tail ratio):
  1. clv_20 / body_ratio_20 / upper_shadow_20 / lower_shadow_20  (intraday price shape)
  2. mom_vol_60 / mom_consistency_20 / rev_5d                     (momentum variants)
  3. btc_beta_60 / us10y_beta_60 / spx_beta_60 / hsi_beta_60      (cross-asset betas)
  4. dd_speed_60 / var_ratio_95_20                                (drawdown / tail shape)
Uses miner3_lib validation pipeline (rank IC vs 10d fwd returns, regime splits,
decay, turnover/coverage, library rho).
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import miner3_lib as L

L.LIB_FACTORS = ['trend_r2_30_signed', 'semi_down_ratio_20', 'mom_120d_skip5',
                 'mom_10d_skip5', 'time_under_water_120', 'vol_of_vol20x60',
                 'dxy_beta_60', 'WTI_BETA_60', 'vix_beta_cond_60x20',
                 'kurt_20', 'tail_ratio_20']

C, V, H, Lw, O = L.load_close_panel(days=4000)
R = C.pct_change()
print(f"panel dates={len(C)} ({C.index.min().date()}..{C.index.max().date()}) assets={C.shape[1]}")

# ---- helpers ----
def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)

def rolling_beta(asset_ret, bench_ret, window):
    """Rolling OLS beta of asset returns on benchmark returns (same window)."""
    cov = asset_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    return safe_div(cov, var)

def drawdown(close):
    return close / close.cummax() - 1.0

def build(name):
    if name == 'clv_20':
        # Close Location Value: where close sits inside day's range, averaged 20d
        rng = (H - Lw).replace(0, np.nan)
        clv = safe_div(C - Lw, rng)
        return clv.rolling(20).mean()
    if name == 'body_ratio_20':
        # candle body dominance: |C-O| / (H-L), averaged 20d
        rng = (H - Lw).replace(0, np.nan)
        body = (C - O).abs()
        return safe_div(body, rng).rolling(20).mean()
    if name == 'upper_shadow_20':
        # upper shadow fraction: (H - max(C,O)) / (H-L)  -- selling pressure at highs
        rng = (H - Lw).replace(0, np.nan)
        us = (H - np.maximum(C, O))
        return safe_div(us, rng).rolling(20).mean()
    if name == 'lower_shadow_20':
        # lower shadow fraction: (min(C,O) - L) / (H-L) -- dip buying support
        rng = (H - Lw).replace(0, np.nan)
        ls = (np.minimum(C, O) - Lw)
        return safe_div(ls, rng).rolling(20).mean()
    if name == 'mom_vol_60':
        # vol-normalized 60d momentum (Sharpe-style trend)
        mom60 = C / C.shift(60) - 1
        vol60 = R.rolling(60).std()
        return safe_div(mom60, vol60)
    if name == 'mom_consistency_20':
        # recency-weighted consistency: mean sign of 5d-forward returns over last 20d
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            sign5 = np.sign(R[s].shift(1).rolling(5).mean())
            out[s] = sign5.rolling(20).mean()
        return out
    if name == 'rev_5d':
        # short-term reversal: negative of 5d return
        return -(C / C.shift(5) - 1).replace([np.inf, -np.inf], np.nan)
    if name == 'btc_beta_60':
        return rolling_beta(R, R['BTC'], 60)
    if name == 'us10y_beta_60':
        return rolling_beta(R, R['US10Y'], 60)
    if name == 'spx_beta_60':
        return rolling_beta(R, R['SPX'], 60)
    if name == 'hsi_beta_60':
        return rolling_beta(R, R['HSI'], 60)
    if name == 'dd_speed_60':
        # drawdown depth per unit of time underwater (how fast/slow the drawdown is)
        dd = drawdown(C)
        tuw = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            # time underwater: days since last peak (rolling count of consecutive dd<0)
            d = dd[s]
            cnt = 0
            vals = []
            for v in d.values:
                if v < 0:
                    cnt += 1
                else:
                    cnt = 0
                vals.append(cnt)
            tuw[s] = vals
        tuw = tuw.replace(0, np.nan)
        return safe_div(dd, tuw)
    if name == 'var_ratio_95_20':
        # 95% historical VaR over 20d scaled by 20d vol: heavy-tail ratio
        var95 = R.rolling(20).quantile(0.05).abs()
        vol20 = R.rolling(20).std()
        return safe_div(var95, vol20)
    raise ValueError(name)

CANDIDATES = ['clv_20', 'body_ratio_20', 'upper_shadow_20', 'lower_shadow_20',
              'mom_vol_60', 'mom_consistency_20', 'rev_5d',
              'btc_beta_60', 'us10y_beta_60', 'spx_beta_60', 'hsi_beta_60',
              'dd_speed_60', 'var_ratio_95_20']

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
       'library_factors': L.LIB_FACTORS, 'results': {}}
for fid in CANDIDATES:
    try:
        panel = build(fid)
        summ = L.full_validate(panel, R, horizon=10, label=fid)
        ic, icir = summ['ic'], summ['icir']
        gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        maxrho = summ['max_abs_library_correlation']
        rho_ok = maxrho < 0.5
        summ['pass_gate'] = bool(gate)
        summ['rho_ok'] = bool(rho_ok)
        out['results'][fid] = {k: summ[k] for k in
                               ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                'regime', 'coverage_asset_days', 'coverage_dates_ge8',
                                'turnover_10d_rank', 'decay_ic_by_horizon',
                                'max_abs_library_correlation', 'pass_gate', 'rho_ok']}
        print(f"{fid:24s} ic={ic:+.4f} icir={icir:+.4f} n={summ['n_ic_dates']:5d} "
              f"cov={summ['coverage_dates_ge8']:.3f} rho={maxrho:.3f} gate={gate} rho_ok={rho_ok}")
    except Exception as e:
        print(f"{fid:24s} ERROR {e}")
        out['results'][fid] = {'error': str(e)}

with open('scripts/miner_1_20271104_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1)
print('\nsaved scripts/miner_1_20271104_explore_results.json')
