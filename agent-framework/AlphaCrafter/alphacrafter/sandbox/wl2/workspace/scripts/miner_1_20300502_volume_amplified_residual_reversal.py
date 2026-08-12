import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
L=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is not None:
  d=d.copy(); d['r1']=d.close.pct_change(); d['r3']=d.close.pct_change(3); d['vol20']=d.r1.rolling(20).std(); d['vrel']=d.volume/d.volume.rolling(20).median(); d['symbol']=s; L.append(d[['date','close','r1','r3','vol20','vrel','symbol']])
z=pd.concat(L).sort_values(['date','symbol']); z['mean3']=z.groupby('date').r3.transform('mean'); z['factor']=-(z.r3-z.mean3)/z.vol20*np.sqrt(z.vrel.clip(.25,4)); z['factor']=z.factor.replace([np.inf,-np.inf],np.nan); z['fwd1']=z.groupby('symbol').close.shift(-1)/z.close-1; z['fwd5']=z.groupby('symbol').close.shift(-5)/z.close-1
def calc(q,col):
 a=[]; ns=[]
 for _,g in q.groupby('date'):
  x=g[['factor',col]].dropna();
  if len(x)>=8:
   c=x.factor.corr(x[col]);
   if np.isfinite(c): a.append(c); ns.append(len(x))
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for h in ['fwd1','fwd5']: print(h,calc(z,h))
for st in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']: print(st,calc(z[z.date>=st],'fwd1'))
print('coverage',z.factor.notna().mean(),'dates',z.date.nunique(),'assets',z.symbol.nunique()); z[['date','symbol','factor']].to_csv('scripts/miner_1_20300502_volume_amplified_residual_reversal_signal.csv',index=False)
