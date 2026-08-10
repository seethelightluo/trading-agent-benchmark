import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
rows=[]
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date')
 d['r1']=d.close.pct_change()
 d['vol20']=d.r1.rolling(20,min_periods=15).std()
 # volatility scaled 1-day residual reversal, with cross-sectional median
 rows.append(d[['date','r1','vol20','close']].assign(symbol=s))
x=pd.concat(rows).sort_values(['date','symbol'])
x['medr']=x.groupby('date')['r1'].transform('median')
x['factor']=-(x.r1-x.medr)/x.vol20
# next close-to-close return, factor observed at t
x['fwd']=x.groupby('symbol').close.pct_change().shift(-1)
# careful group pct_change shift issue: shift within group after pct
x['fwd']=x.groupby('symbol')['close'].transform(lambda z:z.shift(-1)/z-1)
ics=[]; turnover=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','fwd'])
 if len(g)>=8:
  ics.append((dt,spearmanr(g.factor,g.fwd).statistic,len(g)))
# turnover based rank signal changes / all successive common observations
z=x.dropna(subset=['factor']).copy(); z['rank']=z.groupby('date').factor.rank(pct=True)
for s,g in z.groupby('symbol'):
 turnover.extend(np.abs(g['rank'].diff()).dropna().tolist())
a=np.array([v[1] for v in ics]); dates=pd.to_datetime([v[0] for v in ics])
def met(v): return float(np.nanmean(v)), float(np.nanmean(v)/np.nanstd(v,ddof=1))
print('N_DATES',len(a),'AVG_N',np.mean([v[2] for v in ics]),'COVERAGE',len(z)/len(x),'TURNOVER',np.mean(turnover))
print('IC',met(a),'HIT',np.mean(a>0))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=a[(dates>=lo)&(dates<=hi+'-12-31')]; print('REGIME',lo,hi,len(q),met(q))
# horizons from t close to t+h close
for h in [3,5]:
 x['fwdh']=x.groupby('symbol')['close'].transform(lambda z:z.shift(-h)/z-1)
 vals=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor','fwdh'])
  if len(g)>=8: vals.append(spearmanr(g.factor,g.fwdh).statistic)
 print('H',h,met(np.array(vals)))
# artifact
out=x[['date','symbol','factor']].dropna(); out.to_csv('scripts/miner_1_20261217_atr_residual_reversal_signal.csv',index=False)
