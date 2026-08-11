# -*- coding: utf-8 -*-
"""miner_3 2026-12-17: explore novel factor batch (reversal / vol-structure / efficiency).
Visible data through previous completed trading day 2026-12-16.
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe;
post-gate constraint: max_abs_library_correlation < 0.5 (pairwise, artifacts).
"""
import sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
idx = np.arange(len(C))

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
VIX = load_macro('VIX')

def build(name):
    if name == 'rev_5d_xs':
        # cross-sectional short-term reversal: negative 5d return
        return -(C / C.shift(5) - 1.0)
    if name == 'rev_10d_xs':
        return -(C / C.shift(10) - 1.0)
    if name == 'hl_pos_20':
        # position of close inside 20d high-low range
        rng = H.rolling(20).max() - Lw.rolling(20).min()
        return ((C - Lw.rolling(20).min()) / rng - 0.5).replace([np.inf, -np.inf], np.nan)
    if name == 'eff_ratio_20':
        # Kaufman efficiency ratio: net move / path length
        path = R.abs().rolling(20).sum()
        return ((C - C.shift(20)).abs() / path).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ratio_5_60':
        rv5 = R.rolling(5).std(); rv60 = R.rolling(60).std()
        return (rv5 / rv60).replace([np.inf, -np.inf], np.nan)
    if name == 'zscore_60':
        ma = C.rolling(60).mean(); sd = C.rolling(60).std()
        return ((C - ma) / sd).replace([np.inf, -np.inf], np.nan)
    if name == 'intraday_strength_20':
        # mean of close-vs-open return over 20d (intraday buying pressure)
        io = (C / O - 1.0).replace([np.inf, -np.inf], np.nan)
        return io.rolling(20).mean()
    if name == 'corr_cross_20':
        # rolling 20d corr of asset ret to equal-weight cross-section ret
        mkt = R.mean(axis=1)
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(mkt)
        return out
    if name == 'accel_10_60':
        # momentum acceleration: short mom minus long mom
        return (C / C.shift(10) - 1.0) - (C / C.shift(60) - 1.0)
    if name == 'max_dd_20':
        # 20d drawdown depth (negative): how far below 20d high
        return (C / C.rolling(20).max() - 1.0)
    if name == 'rev_vol_cond_5':
        # 5d reversal conditioned on elevated short vol (rv5 > rv60)
        rv5 = R.rolling(5).std(); rv60 = R.rolling(60).std()
        cond = (rv5 > rv60).astype(float)
        return -(C / C.shift(5) - 1.0) * cond
    if name == 'mom_20_calm_vix':
        # 20d momentum active only when VIX below its 60d median (calm regime)
        mom = C / C.shift(20) - 1.0
        calm = (VIX < VIX.rolling(60).median()).astype(float)
        return mom * calm.to_frame('x').reindex(C.index).values if False else mom.mul(calm, axis=0)
    if name == 'beta_ndx_60':
        rv = C['NDX'].pct_change()
        mv = rv.rolling(60).var()
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(60).cov(rv) / mv
        return out.replace([np.inf, -np.inf], np.nan)
    raise ValueError(name)

CANDIDATES = [
    'rev_5d_xs', 'rev_10d_xs', 'hl_pos_20', 'eff_ratio_20', 'vol_ratio_5_60',
    'zscore_60', 'intraday_strength_20', 'corr_cross_20', 'accel_10_60',
    'max_dd_20', 'rev_vol_cond_5', 'mom_20_calm_vix', 'beta_ndx_60',
]

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
       'results': {}}
print(f"Panel: {C.index.min().date()} -> {C.index.max().date()} | {len(C)} dates x {C.shape[1]} assets")
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
                                'library_rho_by_factor', 'max_abs_library_correlation',
                                'pass_gate', 'rho_ok']}
        print(f"[{fid}] IC={ic:.4f} ICIR={icir:.4f} hit={summ['ic_hit_ratio']:.3f} "
              f"n={summ['n_ic_dates']} cov_ge8={summ['coverage_dates_ge8']:.3f} "
              f"to={summ['turnover_10d_rank']:.3f} maxrho={maxrho:.3f} gate={gate} rho_ok={rho_ok}")
        for name, v in summ['regime'].items():
            print(f"     {name}: IC={v['ic']:.4f} ICIR={v['icir']:.4f} n={v['n']}")
        dec = summ['decay_ic_by_horizon']
        print("     decay: " + ", ".join(f"{h}:{icv:.4f}" for h, icv in dec.items()))
        print("     rho_by_factor: " + ", ".join(f"{k}:{v}" for k, v in summ['library_rho_by_factor'].items() if v is not None))
    except Exception as e:
        import traceback; traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner_3_20261217_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved -> scripts/miner_3_20261217_explore_results.json")
