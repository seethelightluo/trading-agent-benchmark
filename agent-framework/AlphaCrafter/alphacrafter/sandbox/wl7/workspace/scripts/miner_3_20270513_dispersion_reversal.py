import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-13')
def f(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):return d.assign(date=pd.to_datetime(d.date).dt.normalize()).drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:f(s) for s in U};D={s:d for s,d in D.items() if d is not None}; C=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); R=C.pct_change();
# High-dispersion conditional short reversal: recent 5d loss is attractive only when lagged cross-sectional dispersion is elevated
r5=C.pct_change(5); disp=R.rolling(10).std().mean(axis=1); gate=(disp>disp.rolling(60).quantile(.65)).astype(float); sig=(-r5/(R.rolling(10).std()*np.sqrt(10)+1e-8)).mul(gate,axis=0).shift(1)
def st(h, pred=lambda d:True):
 fr=C.shift(-h)/C-1; a=[];ns=[]
 for d in sig.index:
  if not pred(d):continue
  z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(a).dropna();return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('assets',len(D),'dates',len(sig),'coverage',sig.notna().sum().sum()/(len(sig)*15));
for h in [1,5,10,20]:print('h',h,st(h))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('reg',a,b,st(1,lambda d:d.year>=a and d.year<=b))
print('turn',sig.rank(axis=1,pct=True).diff().abs().mean().mean());sig.stack().rename('signal').to_csv('scripts/miner_3_20270513_dispersion_reversal_signal.csv')
