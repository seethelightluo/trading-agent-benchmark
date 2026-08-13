"""Screener: market regime + live factor values as of 2032-09-01 (visible through)."""
import numpy as np
import pandas as pd
import json, os

DATA = '../persistent/stock_data'
IDX = '../persistent/index_data'
ASOF = '2032-09-01'
INSTR = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(sym):
    df = pd.read_csv(os.path.join(DATA, sym + '.csv'), parse_dates=[0])
    df.columns = ['date','open','close','high','low','volume','change','pct_change'] + list(df.columns[8:])
    df = df[df['date'] <= ASOF].reset_index(drop=True)
    return df

px = {}
for s in INSTR:
    d = load(s)
    px[s] = d.set_index('date')['close']

px = pd.DataFrame(px).dropna(how='all')
print('rows through', ASOF, ':', len(px), '| last date:', px.index[-1].date())

# Returns
rets = {}
for w in [5, 20, 60, 120, 180, 252]:
    rets[f'r{w}'] = px.iloc[-1] / px.iloc[-1-w] - 1
rdf = pd.DataFrame(rets).T
print('\n=== Cross-sectional returns (through %s) ===' % ASOF)
print(rdf.round(4).to_string())

# flat-feed check: 20d / 60d return == 0 exactly
print('\n=== flat-feed names (20d/60d return ~0) ===')
for s in INSTR:
    if abs(rdf.loc['r20', s]) < 1e-9 and abs(rdf.loc['r60', s]) < 1e-9:
        print(' ', s)

# VIX regime
vix = pd.read_csv(os.path.join(IDX, 'VIX.csv'), parse_dates=[0])
vix.columns = ['date','open','close','high','low','volume','change','pct_change'] + list(vix.columns[8:])
vix = vix[vix['date'] <= ASOF].set_index('date')['close']
print('\n=== VIX (through %s) ===' % ASOF)
print('last:', round(vix.iloc[-1],2), '| 5d ago:', round(vix.iloc[-6],2), '| 20d ago:', round(vix.iloc[-21],2), '| 60d ago:', round(vix.iloc[-61],2))
print('VIX chg 5d: %.1f%% | 20d: %.1f%% | 60d: %.1f%%' % ((vix.iloc[-1]/vix.iloc[-6]-1)*100, (vix.iloc[-1]/vix.iloc[-21]-1)*100, (vix.iloc[-1]/vix.iloc[-61]-1)*100))

# macro signals
for m in ['DXY','USDJPY','USDCNY','EURUSD']:
    mm = pd.read_csv(os.path.join(IDX, m + '.csv'), parse_dates=[0])
    mm.columns = ['date','open','close','high','low','volume','change','pct_change'] + list(mm.columns[8:])
    mm = mm[mm['date'] <= ASOF].set_index('date')['close']
    print('%s: last %.2f | 20d chg %.1f%% | 60d chg %.1f%%' % (m, mm.iloc[-1], (mm.iloc[-1]/mm.iloc[-21]-1)*100, (mm.iloc[-1]/mm.iloc[-61]-1)*100))

# live factor computations (5 admission-gate factors)
print('\n=== LIVE FACTOR VALUES (as of %s) ===' % ASOF)

# 1. mom_180d_skip5 = close.shift(5)/close.shift(185)-1
mom180 = px.iloc[-6] / px.iloc[-186] - 1
print('mom_180d_skip5:'); print(mom180.round(4).sort_values(ascending=False).to_string())

# 2. range_pos_252 = (close - min252)/(max252-min252)
lo252 = px.iloc[-252:].min()
hi252 = px.iloc[-252:].max()
rng = (px.iloc[-1] - lo252) / (hi252 - lo252)
print('\nrange_pos_252:'); print(rng.round(4).sort_values(ascending=False).to_string())

# 3. max_consec_gain_20: longest run of up days in trailing 21d window
r = px.pct_change()
def max_streak(col):
    best = cur = 0
    for v in col.iloc[-21:]:
        if v > 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best
streak = r.apply(max_streak)
print('\nmax_consec_gain_20:'); print(streak.sort_values(ascending=False).to_string())

# 4. downbeta_spx_60: beta of 60d daily returns to SPX on SPX-down days
ret60 = r.iloc[-60:]
spx = ret60['SPX']
mask = spx < 0
downb = {}
for s in INSTR:
    x = spx[mask]; y = ret60[s][mask]
    if len(x) > 5 and y.std() > 0:
        downb[s] = np.cov(x, y)[0,1] / np.var(x)
    else:
        downb[s] = np.nan
print('\ndownbeta_spx_60:'); print(pd.Series(downb).round(4).sort_values(ascending=False).to_string())

# 5. spx_corr60: rolling 60d corr with SPX
corr60 = ret60.corr()['SPX']
print('\nspx_corr60:'); print(corr60.round(4).sort_values(ascending=False).to_string())

# pairwise corr regime
c = r.iloc[-60:].corr()
vals = c.values[np.triu_indices(len(c), 1)]
print('\nmean pairwise |corr| 60d: %.3f | mean corr: %.3f | max |corr|: %.3f' % (np.abs(vals).mean(), vals.mean(), np.abs(vals).max()))

# vol regime: SPX 20d annvol, cross-sectional 20d std
spx20 = px['SPX'].iloc[-21:].pct_change().std() * np.sqrt(252)
cs20 = r.iloc[-20:].std().mean() * np.sqrt(252)
print('SPX 20d annvol: %.1f%% | avg cross-asset 20d annvol: %.1f%%' % (spx20*100, cs20*100))

# 20d cross-sectional dispersion
d20 = px.iloc[-1] / px.iloc[-21] - 1
print('20d cross-sectional std: %.2f%% | spread max-min: %.1fpp' % (d20.std()*100, (d20.max()-d20.min())*100))
