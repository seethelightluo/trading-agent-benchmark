"""miner_1 2027-05-20: explore new factor batch (trend efficiency, range, proximity,
autocorrelation, MACD, asymmetry, vol term structure, zscore, acceleration, cross-asset links).
Uses miner3_lib validation pipeline (rank IC vs 10d fwd returns, regime splits, decay,
turnover/coverage, library rho)."""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import miner3_lib as L

# Extend library factor list to all effective factors with artifacts
L.LIB_FACTORS = ['trend_r2_30_signed', 'semi_down_ratio_20', 'mom_120d_skip5',
                 'mom_10d_skip5', 'time_under_water_120', 'vol_of_vol20x60',
                 'dxy_beta_60', 'WTI_BETA_60', 'vix_beta_cond_60x20',
                 'kurt_20', 'tail_ratio_20']

C, V, H, Lw, O = L.load_close_panel(days=4000)
R = C.pct_change()
# forward return horizon used by full_validate is 10
print(f"panel dates={len(C)} ({C.index.min().date()}..{C.index.max().date()}) assets={C.shape[1]}")

# Macro panels for conditional/cross-asset factors
def load_macro(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv', parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df['close'] if 'close' in df.columns else df.iloc[:, 1]

DXY = load_macro('DXY')
VIX = load_macro('VIX')
USDJPY = load_macro('USDJPY')

EMA12 = C.ewm(span=12, adjust=False).mean()
EMA26 = C.ewm(span=26, adjust=False).mean()
VOL20 = R.rolling(20).std()
VOL5 = R.rolling(5).std()
VOL60 = R.rolling(60).std()

def build(name):
    if name == 'eff_ratio_20':
        # Kaufman efficiency ratio: |C_t - C_{t-20}| / sum(|dC|) over 20d
        num = (C - C.shift(20)).abs()
        den = C.diff().abs().rolling(20).sum()
        return (num / den).replace([np.inf, -np.inf], np.nan)
    if name == 'range_pos_20':
        hi20 = C.rolling(20).max(); lo20 = C.rolling(20).min()
        return ((C - lo20) / (hi20 - lo20)).replace([np.inf, -np.inf], np.nan)
    if name == 'high_prox_60':
        # distance from 60d high: C / rolling_max(H,60) - 1 (<=0, closer to high = bigger)
        hh60 = H.rolling(60).max()
        return (C / hh60 - 1.0).replace([np.inf, -np.inf], np.nan)
    if name == 'ac1_20':
        # lag-1 autocorrelation of daily returns over 20d window
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).apply(lambda x: x.autocorr() if len(x) >= 8 else np.nan, raw=False)
        return out
    if name == 'macd_norm_12_26':
        # (EMA12 - EMA26) normalized by 20d vol
        return ((EMA12 - EMA26) / VOL20).replace([np.inf, -np.inf], np.nan)
    if name == 'gain_loss_20':
        pos = R.clip(lower=0).rolling(20).sum()
        neg = R.clip(upper=0).rolling(20).sum().abs()
        return (pos / neg).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ratio_5_60':
        return (VOL5 / VOL60).replace([np.inf, -np.inf], np.nan)
    if name == 'zscore_20':
        ma20 = C.rolling(20).mean()
        return ((C - ma20) / VOL20).replace([np.inf, -np.inf], np.nan)
    if name == 'accel_10_60':
        # momentum acceleration: 10d return minus 60d return
        return ((C / C.shift(10) - 1) - (C / C.shift(60) - 1)).replace([np.inf, -np.inf], np.nan)
    if name == 'down_vol_ratio_20':
        # downside semi-deviation / total vol (asymmetry)
        neg = R.where(R < 0, 0.0)
        down_vol = (neg.pow(2).rolling(20).mean()).apply(np.sqrt)
        return (down_vol / VOL20).replace([np.inf, -np.inf], np.nan)
    if name == 'xau_corr_20':
        b = R['XAU']
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(b)
        return out
    if name == 'cn_corr_20':
        b = R['000300.SH']
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(b)
        return out
    if name == 'spx_rel_20':
        # relative strength vs SPX: asset 20d return minus SPX 20d return
        spx20 = C['SPX'] / C['SPX'].shift(20) - 1
        return ((C / C.shift(20) - 1).sub(spx20, axis=0)).replace([np.inf, -np.inf], np.nan)
    if name == 'crypto_corr_20':
        b = R['BTC']
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(b)
        return out
    if name == 'vix_level_cond_20':
        # asset 20d momentum conditioned on VIX below its 60d median (risk-on)
        mom20 = C / C.shift(20) - 1
        vix_lo = (VIX < VIX.rolling(60).median()).astype(float)
        return mom20.mul(vix_lo, axis=0)
    if name == 'dxy_ret_cond_20':
        # asset 20d return when DXY fell over 20d (USD weakness risk-on)
        mom20 = C / C.shift(20) - 1
        dxy_down = (DXY.pct_change(20) < 0).astype(float)
        return mom20.mul(dxy_down, axis=0)
    raise ValueError(name)

CANDIDATES = ['eff_ratio_20', 'range_pos_20', 'high_prox_60', 'ac1_20', 'macd_norm_12_26',
              'gain_loss_20', 'vol_ratio_5_60', 'zscore_20', 'accel_10_60', 'down_vol_ratio_20',
              'xau_corr_20', 'cn_corr_20', 'spx_rel_20', 'crypto_corr_20',
              'vix_level_cond_20', 'dxy_ret_cond_20']

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
                                'library_rho_by_factor', 'max_abs_library_correlation',
                                'pass_gate', 'rho_ok']}
        print(f"[{fid}] IC={ic:.4f} ICIR={icir:.4f} hit={summ['ic_hit_ratio']:.3f} "
              f"n={summ['n_ic_dates']} cov_ge8={summ['coverage_dates_ge8']:.3f} "
              f"to={summ['turnover_10d_rank']:.3f} maxrho={maxrho:.3f} gate={gate} rho_ok={rho_ok}")
        for name, v in summ['regime'].items():
            print(f"     {name}: IC={v['ic']:.4f} ICIR={v['icir']:.4f} n={v['n']}")
        dec = summ['decay_ic_by_horizon']
        print("     decay: " + ", ".join(f"{h}:{icv:.4f}" for h, icv in dec.items()))
        rhos = {k: v for k, v in summ['library_rho_by_factor'].items() if v is not None and abs(v) >= 0.4}
        if rhos:
            print("     rho>=0.4: " + ", ".join(f"{k}:{v}" for k, v in rhos.items()))
    except Exception as e:
        import traceback; traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner_1_20270520_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved -> scripts/miner_1_20270520_explore_results.json")
