import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-13')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None};rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change();
 # cross-sectional residual momentum: own 20d return minus universe mean, volatility normalized; fully lagged
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'ret20':c.pct_change(20),'vol':r.rolling(20).std()*np.sqrt(20)}))
q=pd.concat(rows,ignore_index=True).pivot(index='date',columns='asset',values=['ret20','vol'])
res=q['ret20'].sub(q['ret20'].mean(axis=1),axis=0); sig=(-res/(q['vol']+1e-8)).shift(1)
close=pd.concat({s:D[s].close.astype(float) for s in D},axis=1).sort_index()
def stats(h, subset=None):
 fr=close.shift(-h)/close-1; v=[]; n=[]
 for dt in sig.index:
  if subset is not None and not subset(dt):continue
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
 a=pd.Series(v).dropna();return len(a),np.mean(n),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('assets',len(D),'dates',sig.index.nunique(),'coverage',sig.notna().sum().sum()/(sig.shape[0]*15))
for h in [1,5,10,20]:print('horizon',h,stats(h))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,stats(1,lambda d:d.year>=a and d.year<=b))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
sig.stack().rename('signal').to_csv('scripts/miner_3_20270513_residual_momentum_signal.csv')
