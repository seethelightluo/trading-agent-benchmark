# -*- coding: utf-8 -*-
"""miner_3 2028-03-23: explore novel factor batch A (data through 2028-03-22).
Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon on the 15-asset universe.
Robustness sanity: n_ic_dates >= 120 and coverage_dates_ge8 >= 0.6.
All rolling computations vectorized (no per-date/per-column python loops).
Batch A: trend quality, extremes asymmetry, vol term structure, systematic-ness,
downside/China/rate/JPY/DXY-up betas.
"""
import sys, json, os, io, zlib, base64, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# ---- real effective library (with artifacts) for rho check ----
lib_factors = []
for p in sorted(os.listdir('factors')):
    if not p.endswith('.json') or p == 'factor_ensemble.json':
        continue
    try:
        d = json.load(open('factors/' + p))
        if d.get('validation', {}).get('status') == 'EFFECTIVE' and \
           d.get('validation', {}).get('signal_artifact'):
            lib_factors.append(d['factor_id'])
    except Exception:
        pass
print('Library factors for rho check (%d): %s' % (len(lib_factors), lib_factors))

# ---- cached decoded library panels ----
LIB_PANELS = {}
for fid in lib_factors:
    try:
        d = json.load(open('factors/%s.json' % fid))
        art = d.get('validation', {}).get('signal_artifact')
        if art:
            raw = base64.b64decode(art['data'])
            df = pd.read_csv(io.StringIO(zlib.decompress(raw).decode('utf-8')), index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            LIB_PANELS[fid] = df
    except Exception as e:
        print('  rho-cache skip %s: %s' % (fid, e))

def max_lib_rho(new_panel):
    rhos = {}
    new_flat = new_panel.stack()
    for fid, libp in LIB_PANELS.items():
        common = new_panel.index.intersection(libp.index)
        a = new_panel.loc[common].stack()
        b = libp.loc[common].stack()
        m = a.notna() & b.notna()
        if m.sum() >= 200:
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            rhos[fid] = round(r, 3) if not math.isnan(r) else None
        else:
            rhos[fid] = None
    vals = [v for v in rhos.values() if v is not None]
    return rhos, (max(vals) if vals else 0.0)

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
USDJPY = load_macro('USDJPY')

def rolling_beta(x, f, win):
    cov = x.rolling(win).cov(f)
    var = f.rolling(win).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

def rolling_beta_cond(x, f, win, cond):
    xc = x.where(cond)
    fc = f.where(cond)
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        m = xc[s].notna() & fc.notna()
        out[s] = xc[s].rolling(win, min_periods=15).cov(fc) / fc.rolling(win, min_periods=15).var()
    return out.replace([np.inf, -np.inf], np.nan)

def rolling_corr_with(x, f, win):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        out[s] = x[s].rolling(win).corr(f)
    return out.replace([np.inf, -np.inf], np.nan)

def build(name):
    if name == 'eff_ratio_20_signed':
        path = R.abs().rolling(20).sum()
        net = (C - C.shift(20)).abs()
        eff = (net / path).replace([np.inf, -np.inf], np.nan)
        return eff * np.sign(C.pct_change(20))
    if name == 'max_gain_loss_20':
        g = R.rolling(20).max()
        l = R.rolling(20).min()
        return (g / l.abs()).replace([np.inf, -np.inf], np.nan)
    if name == 'updown_ratio_60':
        pos = R.where(R > 0, np.nan).rolling(60, min_periods=20).mean()
        neg = R.where(R < 0, np.nan).rolling(60, min_periods=20).mean()
        return (pos / neg.abs()).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ts_5_60':
        return (R.rolling(5).std() / R.rolling(60).std()).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ts_10_60':
        return (R.rolling(10).std() / R.rolling(60).std()).replace([np.inf, -np.inf], np.nan)
    if name == 'uni_corr_60':
        f = R.mean(axis=1)
        return rolling_corr_with(R, f, 60)
    if name == 'downside_beta_60':
        f = R.mean(axis=1)
        return rolling_beta_cond(R, f, 60, f < 0)
    if name == 'chn_beta_60':
        f = R['000300.SH']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    if name == 'rate_beta_60':
        f = R['US10Y']
        return pd.DataFrame({s: rolling_beta(R[s], f, 60) for s in R.columns}, index=R.index)
    if name == 'jpy_beta_60':
        fj = USDJPY.pct_change()
        return pd.DataFrame({s: rolling_beta(R[s], fj, 60) for s in R.columns}, index=R.index)
    if name == 'dxy_up_beta_60':
        fd = DXY.pct_change()
        return rolling_beta_cond(R, fd, 60, fd > 0)
    return None

def summarize(s, horizon, label=''):
    ic = s.mean()
    icir = s.mean() / s.std() if s.std() > 0 else 0.0
    hit = (s > 0).mean()
    regime = {}
    for name, lo, hi in [('2020-2022', '2020-01-01', '2022-12-31'),
                          ('2023-2024', '2023-01-01', '2024-12-31'),
                          ('2025-2026', '2025-01-01', '2026-12-31'),
                          ('2027-2028', '2027-01-01', '2028-12-31')]:
        sub = s[(s.index >= lo) & (s.index <= hi)]
        if len(sub) >= 20:
            regime[name] = {'ic': round(sub.mean(), 4),
                            'icir': round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0,
                            'n': int(len(sub))}
    return {'label': label, 'horizon': horizon, 'ic': round(ic, 4), 'icir': round(icir, 4),
            'ic_hit_ratio': round(hit, 3), 'n_ic_dates': int(len(s)), 'regime': regime}

def fast_validate(fp, horizon=10, label=''):
    s = L.rank_ic(fp, R.shift(-horizon))
    summ = summarize(s, horizon, label)
    decay = {}
    for h in (5, 10, 20):
        sh = L.rank_ic(fp, R.shift(-h))
        if sh is not None and len(sh) >= 20:
            decay[str(h)] = round(sh.mean(), 4)
    summ['decay_ic_by_horizon'] = decay
    summ.update(L.coverage_turnover(fp, R, horizon))
    rhos, maxrho = max_lib_rho(fp)
    summ['max_abs_library_correlation'] = round(maxrho, 3)
    return summ

CANDIDATES = ['eff_ratio_20_signed', 'max_gain_loss_20', 'updown_ratio_60',
              'vol_ts_5_60', 'vol_ts_10_60', 'uni_corr_60', 'downside_beta_60',
              'chn_beta_60', 'rate_beta_60', 'jpy_beta_60', 'dxy_up_beta_60']

results = {}
for name in CANDIDATES:
    fp = build(name)
    if fp is None:
        print('\n[%s] build failed' % name)
        continue
    try:
        summ = fast_validate(fp, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    n_ok = summ['n_ic_dates'] >= 120
    cov_ok = summ['coverage_dates_ge8'] >= 0.6
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    results[name] = summ
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  max_abs_library_corr=%.3f' % summ['max_abs_library_correlation'])
    print('  GATE: ic=%s icir=%s robust(n>=120,cov>=0.6)=%s => PASS=%s'
          % (gate_ic, gate_icir, robust := (n_ok and cov_ok), gate_ic and gate_icir and robust))

out = {'visible_through': str(C.index.max().date()), 'n_dates': int(len(C)),
       'n_assets': int(C.shape[1]), 'library_factors': lib_factors, 'results': results}
with open('scripts/miner_3_20280323_explore_results_A.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_3_20280323_explore_results_A.json')
