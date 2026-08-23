import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=3000) for s in U}; macro=get_index_daily_data('DXY',days=3000)
close=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in px.items() if d is not None and len(d)}).sort_index()
m=macro.set_index('date')['close'].astype(float)
r5=close.pct_change(5); mr=m.pct_change(5)
shock=(mr-mr.rolling(252,min_periods=60).median()).clip(lower=0).clip(upper=.05)/.05
sig=(-r5).mul(shock.reindex(close.index).fillna(0),axis=0)
ics=[]; counts=[]; turnovers=[]; prev=None
for d in close.index:
 z=pd.concat([sig.loc[d],(close.shift(-10)/close-1).loc[d]],axis=1).dropna()
 if len(z)>=8 and sig.loc[d].abs().sum()>0:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append((d,c)); counts.append(len(z))
 rank=sig.loc[d].rank(pct=True)
 if prev is not None:
  q=pd.concat([rank,prev],axis=1).dropna()
  if len(q): turnovers.append(float((q.iloc[:,0]-q.iloc[:,1]).abs().mean()))
 prev=rank
iv=np.array([x[1] for x in ics]); dates=np.array([x[0] for x in ics])
def sm(a):
 sd=np.std(a,ddof=1) if len(a)>1 else 0
 return np.mean(a),np.mean(a)/sd if sd else 0,np.mean(a>0),len(a)
print('cutoff',close.index.max().date(),'dates',len(iv),'avg_inst',np.mean(counts),'coverage',np.mean(counts)/15,'turnover',np.mean(turnovers))
print('10d',sm(iv))
for n in [180,360]: print('recent',n,sm(iv[dates>=dates[-1]-pd.tseries.offsets.BDay(n)]))
for yr in sorted(set(d.year for d in dates)): print('year',yr,sm(iv[dates.astype("datetime64[Y]").astype(int)+1970==yr]))
for h in [5,20]:
 ff=close.shift(-h)/close-1; aa=[]
 for d in close.index:
  z=pd.concat([sig.loc[d],ff.loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,sm(np.array(aa)))
sig.index.name='date'; sig.to_csv('scripts/miner_2_20310403_dxy_shock_reversal_signal.csv')
