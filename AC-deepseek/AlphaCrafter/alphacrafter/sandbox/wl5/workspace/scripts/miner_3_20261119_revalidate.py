# -*- coding: utf-8 -*-
"""miner_3 2026-11-19: periodic re-validation of all currently EFFECTIVE factors (fixed).
Visible data through previous completed trading day (2026-11-18).
Gates: |IC|>=0.007, |ICIR|>=0.084 @10d horizon."""
import sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
VIX = load_macro('VIX')

def beta_to(panel_ret, macro_ret, win):
    out = pd.DataFrame(index=panel_ret.index, columns=panel_ret.columns, dtype=float)
    mv = macro_ret.rolling(win).var()
    for s in panel_ret.columns:
        out[s] = panel_ret[s].rolling(win).cov(macro_ret) / mv
    return out.replace([np.inf, -np.inf], np.nan)

def trend_r2_30_signed():
    idx = np.arange(len(C))
    out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        x = np.log(C[s])
        win = 30
        vals = [np.nan] * (win - 1)
        for i in range(win - 1, len(x)):
            y = x.iloc[i - win + 1:i + 1]
            if y.notna().sum() < 18:
                vals.append(np.nan)
                continue
            yy = y.dropna().values
            tt = idx[i - len(yy) + 1:i + 1].astype(float)
            tt = tt - tt.mean()
            cov = np.cov(tt, yy)[0, 1]
            var_t = tt.var()
            var_y = yy.var()
            r2 = cov * cov / (var_t * var_y) if var_t > 0 and var_y > 0 else np.nan
            vals.append(np.sign(cov) * r2)
        out[s] = vals
    return out

def semi_down_ratio_20():
    dn = np.sqrt((np.minimum(R, 0) ** 2).rolling(20).mean())
    up = np.sqrt((np.maximum(R, 0) ** 2).rolling(20).mean())
    return (dn / up - 1.0).replace([np.inf, -np.inf], np.nan)

def mom_120d_skip5():
    return C.shift(5) / C.shift(125) - 1.0

def mom_10d_skip5():
    return C.shift(5) / C.shift(15) - 1.0

def vol_of_vol20x60():
    rv20 = R.rolling(20).std()
    return rv20.rolling(60).std()

def dxy_beta_60():
    return beta_to(R, DXY.pct_change(), 60)

def time_under_water_120():
    rollmax = C.rolling(120).max()
    out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        rm = rollmax[s]
        tuw = np.nan
        vals = []
        for i in range(len(C)):
            if pd.isna(rm.iloc[i]):
                vals.append(np.nan); continue
            if C[s].iloc[i] >= rm.iloc[i] - 1e-12:
                tuw = 0
            elif not pd.isna(tuw):
                tuw += 1
            vals.append(tuw)
        out[s] = vals
    return out

def kurt_20():
    return R.rolling(20, min_periods=8).kurt()

def WTI_BETA_60():
    return beta_to(R, C['WTI'].pct_change(), 60)

def vix_beta_cond_60x20():
    rv = VIX.pct_change()
    beta = beta_to(R, rv, 60)
    vix_chg = VIX / VIX.shift(20) - 1.0
    return -beta * vix_chg

def btc_beta_60():
    return beta_to(R, C['BTC'].pct_change(), 60)

def eth_beta_60():
    return beta_to(R, C['ETH'].pct_change(), 60)

BUILDERS = {
    'trend_r2_30_signed': trend_r2_30_signed,
    'semi_down_ratio_20': semi_down_ratio_20,
    'mom_120d_skip5': mom_120d_skip5,
    'mom_10d_skip5': mom_10d_skip5,
    'vol_of_vol20x60': vol_of_vol20x60,
    'dxy_beta_60': dxy_beta_60,
    'time_under_water_120': time_under_water_120,
    'kurt_20': kurt_20,
    'WTI_BETA_60': WTI_BETA_60,
    'vix_beta_cond_60x20': vix_beta_cond_60x20,
    'btc_beta_60': btc_beta_60,
    'eth_beta_60': eth_beta_60,
}

out = {'visible_through': str(C.index.max().date()), 'results': {}}
for fid, fn in BUILDERS.items():
    try:
        panel = fn()
        summ = L.full_validate(panel, R, horizon=10, label=fid)
        ic, icir = summ['ic'], summ['icir']
        gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        summ['pass_gate'] = bool(gate)
        out['results'][fid] = {k: summ[k] for k in
                               ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                'regime', 'coverage_asset_days', 'coverage_dates_ge8',
                                'turnover_10d_rank', 'decay_ic_by_horizon',
                                'max_abs_library_correlation', 'pass_gate']}
        print(f"[{fid}] IC={ic:.4f} ICIR={icir:.4f} hit={summ['ic_hit_ratio']:.3f} n={summ['n_ic_dates']} "
              f"cov_ge8={summ['coverage_dates_ge8']:.3f} to={summ['turnover_10d_rank']:.3f} "
              f"maxrho={summ['max_abs_library_correlation']:.3f} gate={gate}")
        for name, v in summ['regime'].items():
            print(f"     {name}: IC={v['ic']:.4f} ICIR={v['icir']:.4f} n={v['n']}")
        dec = summ['decay_ic_by_horizon']
        print(f"     decay: " + ", ".join(f"{h}:{icv:.4f}" for h, icv in dec.items()))
    except Exception as e:
        import traceback; traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner_3_20261119_revalidate_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved results to scripts/miner_3_20261119_revalidate_results.json")
