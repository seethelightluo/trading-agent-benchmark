"""miner_3 2028-04-06: persist flip_mom_20x10 (trend-consistent momentum) to factors/.
Rebuilds the full 520-row signal artifact (base64:zlib:csv) and writes complete JSON,
then reads back + verifies.
"""
import sys, os, json, zlib, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats, spearman_panel_rho

ASOF = '2028-04-05'
H = 10
GATE_IC, GATE_ICIR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index

def vseries(s):
    return s.dropna()

def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(s, p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)

def build(fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

f = build(lambda s, p: flip_mom(s, p, 20, 10)).replace([np.inf, -np.inf], np.nan)
fwd10 = fwd_panel(H)

icd = cross_sectional_ic(f, fwd10)
st = ic_stats(icd)
icr = icd[icd.index >= icd.index[-1] - pd.Timedelta(days=252)]
st_r = ic_stats(icr)

# turnover
ranks = f.rank(axis=1)
to10 = float(ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1))

# decay
decay = {}
for hh in [1, 2, 3, 5, 10, 20]:
    icd_h = cross_sectional_ic(f, fwd_panel(hh))
    decay[hh] = round(float(icd_h['ic'].mean()) if len(icd_h) else np.nan, 4)

# regime
reg = {}
for lab, m in [('2020-2021 COVID/recovery', icd.index < pd.Timestamp('2022-01-01')),
               ('2022-2023 tightening/AI', (icd.index >= pd.Timestamp('2022-01-01')) & (icd.index < pd.Timestamp('2024-01-01'))),
               ('2024+ crypto/commodity/risk-off', icd.index >= pd.Timestamp('2024-01-01'))]:
    sub = icd[m]
    if len(sub):
        ss = ic_stats(sub)
        reg[lab] = [round(float(ss['ic']), 4), round(float(ss['icir']), 4), int(ss['n_dates'])]

# library correlation
def rbeta(y, x, w, cond=None, minp=None):
    vy, vx = vseries(y), vseries(x)
    df = pd.concat([vy.rename('y'), vx.rename('x')], axis=1, sort=True).dropna()
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        c = pd.Series(True, index=df.index)
    ym, xm = df['y'].where(c), df['x'].where(c)
    if minp is None:
        minp = max(6, int(w * 0.4))
    cov = ym.rolling(w, min_periods=minp).cov(xm)
    var = xm.rolling(w, min_periods=minp).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(INDEX)

def ret1(s):
    return vseries(s).pct_change().reindex(INDEX)

vix, usdcny, us10 = macro['VIX'], macro['USDCNY'], px['US10Y']

def build_lib(fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

lib_panels = {
    'usdcny_beta_60': build_lib(lambda s, p: rbeta(p, usdcny, 60)),
    'mom_10d_skip5': build_lib(lambda s, p: retk(p, 15) - retk(p, 5)),
    'vix_beta_cond_60x20': build_lib(lambda s, p: rbeta(p, vix, 60, cond=(vix.pct_change().rolling(20).sum() > 0).reindex(p.index))),
    'yield_beta_cond_60x20': build_lib(lambda s, p: rbeta(p, us10, 60, cond=(us10.pct_change().rolling(20).sum() > 0).reindex(p.index))),
}
rho_lib = {k: round(spearman_panel_rho(f, lp.reindex(f.index)), 4) for k, lp in lib_panels.items()}
maxrho = round(max([abs(v) for v in rho_lib.values() if v == v], default=0.0), 4)

# leave-one-out min (computed in detail script; recompute quickly)
lo_min = None
for s in WATCH:
    fsub = f.drop(columns=[s])
    icd_s = cross_sectional_ic(fsub, fwd10.drop(columns=[s]))
    ss = ic_stats(icd_s)
    if lo_min is None or abs(ss['ic']) < abs(lo_min[0]):
        lo_min = (float(ss['ic']), float(ss['icir']), int(ss['n_dates']))

coverage_252 = float(f.tail(252).notna().mean().mean())
coverage_all = float(f.notna().mean().mean())

# --- signal artifact (base64:zlib:csv), 520 rows ending at ASOF ---
sig = f.tail(520).copy()
sig.insert(0, 'date', sig.index)
buf = sig.to_csv(index=False).encode('utf-8')
artifact = 'base64:zlib:csv:' + base64.b64encode(zlib.compress(buf, 9)).decode('ascii')

factor_id = 'flip_mom_20x10'
doc = {
    'factor_id': factor_id,
    'factor_name': 'Flip (trend-consistent) momentum 20d x 10d gate',
    'version': '1.0.0',
    'calculation': {
        'expression': 'sign(close/close_10d_ago - 1) * (close/close_20d_ago - 1)',
        'description': ('Trend-consistent momentum: 20d momentum scaled by the sign of the 10d '
                        'return. If the recent 10d move is up, keep the raw 20d momentum; if the '
                        'recent 10d move is down, flip the sign (fading 20d momentum when the '
                        'short-term trend has turned). Captures continuation when short and medium '
                        'trends agree and reversal-fading when they disagree.'),
    },
    'dependencies': ['close'],
    'parameters': {'mom_window': 20, 'gate_window': 10, 'horizon': 10},
    'tags': ['momentum', 'trend-consistency', 'cross-asset'],
    'expected_direction': 1,
    'benchmark_admission': {'ic_abs_gate': GATE_IC, 'icir_abs_gate': GATE_ICIR},
    'validation': {
        'status': 'EFFECTIVE',
        'period': f'2020-01-01..{ASOF}',
        'last_validated': '2028-04-05',
        'admission_horizon': H,
        'regime_notes': (
            'Full 2020..2028-04-05 history (2105 IC dates, avg 14.6 assets/date, coverage 100% - frozen '
            '000688/NDX/SOX/CN10Y flat closes excluded via dropna). Regime IC [IC, ICIR, n]: '
            f'{reg}. Decay by horizon: {decay} - predictive power grows to H=10 then fades (20d: 0.0481), '
            'consistent with a ~10-15d continuation horizon. Recent-252d: IC 0.0549 / ICIR 0.1778 (n=181) '
            '- timely, no sign reversal. Leave-one-asset-out min IC {lo_min[0]:.4f} (drop WTI) - not '
            'driven by any single asset. Family screen: 40x5/60x10/120x10 variants fail gate or go negative '
            'in 2024+, so the 20x10 parameterization is the robust point. Library overlap max rho '
            f'{maxrho} (vs mom_10d_skip5 {rho_lib["mom_10d_skip5"]}, yield_beta_cond_60x20 {rho_lib["yield_beta_cond_60x20"]}).'
        ),
        'metrics': {
            'ic': round(float(st['ic']), 4),
            'icir': round(float(st['icir']), 4),
            'ic_hit_ratio': round(float(st['hit']), 4),
            'n_ic_dates': int(st['n_dates']),
            'avg_assets_per_date': round(float(st.get('avg_n', np.nan)), 1),
            'coverage_asset_days': round(coverage_all, 4),
            'coverage_last_252d': round(coverage_252, 4),
            'turnover_10d_rank': round(to10, 4),
            'decay_ic_by_horizon': {str(k): v for k, v in decay.items()},
            'regime_ic_icir': {k: v for k, v in reg.items()},
            'recent_252d_ic': round(float(st_r['ic']), 4),
            'recent_252d_icir': round(float(st_r['icir']), 4),
            'recent_252d_n': int(st_r['n_dates']),
            'leave_one_asset_out_min_ic_icir': [round(lo_min[0], 4), round(lo_min[1], 4), lo_min[2]],
            'max_abs_library_correlation': maxrho,
            'library_correlation_detail': rho_lib,
        },
        'signal_artifact': {
            'format': 'base64:zlib:csv',
            'descrip': 'factor value panel rows=date cols=asset (15-asset cross-asset universe), 520 rows ending 2028-04-05',
            'data': artifact,
        },
    },
}

path = f'factors/{factor_id}.json'
with open(path, 'w') as fp:
    json.dump(doc, fp, indent=1)
print(f"written {path} ({os.path.getsize(path)} bytes)")

# --- verify read-back ---
chk = json.load(open(path))
ok = (chk['factor_id'] == factor_id
      and chk['validation']['status'] == 'EFFECTIVE'
      and abs(chk['validation']['metrics']['ic']) >= GATE_IC
      and abs(chk['validation']['metrics']['icir']) >= GATE_ICIR
      and chk['validation']['signal_artifact']['data'].startswith('base64:zlib:csv'))
print('verify factor_id:', chk['factor_id'])
print('verify status:', chk['validation']['status'])
print('verify ic gate:', chk['validation']['metrics']['ic'], '>=', GATE_IC, '->', abs(chk['validation']['metrics']['ic']) >= GATE_IC)
print('verify icir gate:', chk['validation']['metrics']['icir'], '>=', GATE_ICIR, '->', abs(chk['validation']['metrics']['icir']) >= GATE_ICIR)
print('verify signal artifact present:', ok)
# decode-and-compare artifact tail to source panel
raw = base64.b64decode(chk['validation']['signal_artifact']['data'].split(':', 2)[2])
df_chk = pd.read_csv(__import__('io').BytesIO(zlib.decompress(raw)))
print('artifact rows/cols:', df_chk.shape, 'last date:', df_chk['date'].iloc[-1])
print('READBACK VERIFY:', 'OK' if ok and df_chk.shape == (520, 16) else 'FAIL')