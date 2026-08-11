"""miner_2: validate downside/upside volatility asymmetry factor (fixed min_periods).

Variants:
  A) ratio std(r|r<0)/std(r|r>0), 20d/60d windows, min_periods=10
  B) semi-deviation ratio: sqrt(mean(min(r,0)^2)) / sqrt(mean(max(r,0)^2)) over full window
  C) downside minus upside semi-deviation (signed asymmetry)
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           build_library_panels, max_library_correlation)

prices = load_prices(days=2100)
lib_panels = build_library_panels(prices)


def f_cond_vol_ratio(df, s, window=60, log=False):
    r = df['close'].pct_change()
    neg = r.where(r < 0)
    pos = r.where(r > 0)
    sdn = neg.rolling(window, min_periods=10).std()
    sdp = pos.rolling(window, min_periods=10).std()
    ratio = sdn / sdp
    if log:
        ratio = np.log(ratio)
    return ratio.reindex(r.index)


def f_semidev_ratio(df, s, window=60, log=False):
    r = df['close'].pct_change()
    neg2 = r.clip(upper=0) ** 2
    pos2 = r.clip(lower=0) ** 2
    sdn = neg2.rolling(window, min_periods=window // 2).mean().apply(np.sqrt)
    sdp = pos2.rolling(window, min_periods=window // 2).mean().apply(np.sqrt)
    ratio = sdn / sdp
    if log:
        ratio = np.log(ratio)
    return ratio.reindex(r.index)


for window in (20, 60):
    for log in (False, True):
        fid = f'du_cond_vol_{window}' + ('_log' if log else '')
        panel = factor_to_panel(lambda df, s, w=window, lg=log: f_cond_vol_ratio(df, s, w, lg), prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient -> None')
            continue
        rho, fid2 = max_library_correlation(panel, lib_panels)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = fid2
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f'{fid}: panel {panel.shape} | IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} '
              f'hit={m["ic_hit_ratio"]:.3f} cov={m["coverage_asset_days"]:.3f} '
              f'ge8={m["coverage_dates_ge8"]:.3f} turn={m["turnover_10d_rank"]:.3f} '
              f'rho={rho:.3f}({fid2}) -> {"PASS" if ok else "FAIL"}')
        print('   decay:', {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})

for window in (20, 60):
    for log in (False, True):
        fid = f'du_semidev_{window}' + ('_log' if log else '')
        panel = factor_to_panel(lambda df, s, w=window, lg=log: f_semidev_ratio(df, s, w, lg), prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient -> None')
            continue
        rho, fid2 = max_library_correlation(panel, lib_panels)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = fid2
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f'{fid}: panel {panel.shape} | IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} '
              f'hit={m["ic_hit_ratio"]:.3f} cov={m["coverage_asset_days"]:.3f} '
              f'ge8={m["coverage_dates_ge8"]:.3f} turn={m["turnover_10d_rank"]:.3f} '
              f'rho={rho:.3f}({fid2}) -> {"PASS" if ok else "FAIL"}')
        print('   decay:', {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
