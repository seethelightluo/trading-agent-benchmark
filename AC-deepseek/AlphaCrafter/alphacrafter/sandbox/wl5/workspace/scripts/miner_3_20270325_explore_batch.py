# -*- coding: utf-8 -*-
"""miner_3 2027-03-25: explore novel factor batch (vol-structure / return-asymmetry /
autocorrelation / liquidity / macro-conditional momentum / cross-asset correlation).
Visible data through previous completed trading day 2027-03-24.
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe;
post-gate constraint: max_abs_library_correlation < 0.5 (pairwise, artifacts).
Data quirks (verified): volume == 0 for SOX/XAU/COPPER/WTI/US10Y/CN10Y -> volume
factors have ~9/15 asset coverage; high/low missing for SOX/US10Y/CN10Y -> avoid
high/low-based factors.
"""
import sys, json, math, glob, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# --- dynamic library list: all EFFECTIVE factors with recoverable signal artifacts ---
lib_factors = []
for p in sorted(glob.glob('factors/*.json')):
    if os.path.basename(p) == 'factor_ensemble.json':
        continue
    try:
        d = json.load(open(p))
        if d.get('validation', {}).get('status') == 'EFFECTIVE' and \
           d.get('validation', {}).get('signal_artifact'):
            lib_factors.append(d['factor_id'])
    except Exception:
        pass
L.LIB_FACTORS = lib_factors
print('Library factors for rho check (%d): %s' % (len(lib_factors), lib_factors))

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
VIX = load_macro('VIX')
print(f"Panel: {C.index.min().date()} -> {C.index.max().date()} | {len(C)} dates x {C.shape[1]} assets")

def build(name):
    # ---- vol structure ----
    if name == 'vol_mom_20_60':
        rv20 = R.rolling(20).std(); rv60 = R.rolling(60).std()
        return (rv20 / rv60 - 1.0).replace([np.inf, -np.inf], np.nan)
    if name == 'up_down_ratio_20':
        up = R.where(R > 0); dn = R.where(R < 0)
        upm = up.rolling(20).mean(); dnm = dn.rolling(20).mean().abs()
        return (upm / dnm).replace([np.inf, -np.inf], np.nan)
    if name == 'pos_day_ratio_20':
        return (R > 0).astype(float).rolling(20).mean()
    if name == 'ac1_30':
        # lag-1 autocorrelation of daily returns over trailing 30d (trend persistence)
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = R[s]
            out[s] = x.rolling(30).corr(x.shift(1))
        return out
    if name == 'cvar_20':
        # mean of the worst 5 daily returns over 20d (tail magnitude, negative)
        return R.rolling(20).apply(lambda a: np.mean(np.sort(a)[:5]) if len(a) >= 20 else np.nan, raw=True)
    if name == 'mom_vol_adj_20':
        rv20 = R.rolling(20).std()
        return ((C / C.shift(20) - 1.0) / rv20).replace([np.inf, -np.inf], np.nan)
    if name == 'rev_10_vol_20':
        # short-term reversal scaled by vol: high-vol assets reverse more
        rv20 = R.rolling(20).std()
        return (-(C / C.shift(10) - 1.0) * rv20).replace([np.inf, -np.inf], np.nan)
    # ---- liquidity (9-asset coverage) ----
    if name == 'volume_z_20':
        v20 = V.rolling(20).mean(); v60 = V.rolling(60).mean(); sd60 = V.rolling(60).std()
        return ((v20 - v60) / sd60).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_confirm_mom_20':
        v20 = V.rolling(20).mean(); v60 = V.rolling(60).mean(); sd60 = V.rolling(60).std()
        vz = ((v20 - v60) / sd60).replace([np.inf, -np.inf], np.nan)
        mom20 = C / C.shift(20) - 1.0
        return (mom20 * np.sign(vz)).replace([np.inf, -np.inf], np.nan)
    # ---- macro-conditional ----
    if name == 'mom_20_dxy_cond':
        mom20 = C / C.shift(20) - 1.0
        dxy_down = (DXY.pct_change(20) < 0).astype(float)
        return mom20.mul(dxy_down, axis=0)
    if name == 'risk_on_mom_20':
        mom20 = C / C.shift(20) - 1.0
        spx = C['SPX']
        risk_on = (spx > spx.rolling(20).mean()).astype(float)
        return mom20.mul(risk_on, axis=0)
    if name == 'mom_20_low_disp':
        mom20 = C / C.shift(20) - 1.0
        disp = mom20.std(axis=1)  # cross-sectional dispersion of 20d returns
        low = (disp < disp.rolling(60).median()).astype(float)
        return mom20.mul(low, axis=0)
    if name == 'breadth_mom_10':
        mom10 = C / C.shift(10) - 1.0
        breadth = (mom10 > 0).mean(axis=1)  # fraction of assets up over 10d
        hi = (breadth > 0.5).astype(float)
        return mom10.mul(hi, axis=0)
    # ---- cross-asset correlation ----
    if name == 'corr_bond_stock_20':
        # rolling 20d corr of each asset's return with US10Y return
        b = R['US10Y']
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(b)
        return out
    if name == 'dxy_corr_20':
        d = DXY.pct_change()
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            out[s] = R[s].rolling(20).corr(d)
        return out
    raise ValueError(name)

CANDIDATES = [
    'vol_mom_20_60', 'up_down_ratio_20', 'pos_day_ratio_20', 'ac1_30', 'cvar_20',
    'mom_vol_adj_20', 'rev_10_vol_20', 'volume_z_20', 'vol_confirm_mom_20',
    'mom_20_dxy_cond', 'risk_on_mom_20', 'mom_20_low_disp', 'breadth_mom_10',
    'corr_bond_stock_20', 'dxy_corr_20',
]

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
       'library_factors': lib_factors, 'results': {}}
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

with open('scripts/miner_3_20270325_explore_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved -> scripts/miner_3_20270325_explore_results.json")
