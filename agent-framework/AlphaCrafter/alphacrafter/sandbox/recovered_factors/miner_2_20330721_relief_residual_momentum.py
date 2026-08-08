import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); return d.close.astype(float)
p=pd.DataFrame({a:load(a) for a in A}).sort_index(); r=np.log(p).diff()
def macro(a):
 d=pd.read_csv('../persistent/index_data/'+a+'.csv',parse_dates=['date']).set_index('date'); return np.log(d.close.astype(float)).diff().reindex(p.index).ffill()
v=macro('VIX'); d=macro('DXY')
# In a simultaneous risk-relief regime, rank assets by 40d residual momentum
# versus global equal-weight benchmark; lag the conditional signal one day.
bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
sig=(resid.rolling(40,min_periods=30).sum().sub(resid.rolling(40,min_periods=30).sum().mean(axis=1),axis=0) if False else resid.rolling(40,min_periods=30).sum())
gate=(v.rolling(5).sum()<0)&(d.rolling(5).sum()<0)
sig=sig.where(gate[:,None] if False else gate, np.nan).shift(1)
print('range',p.index.min(),p.index.max(),'assets',len(A),'gate_fraction',gate.mean())
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8 and sig.loc[dt,ok].nunique()>=3:
   q=spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic
   if np.isfinite(q): vals.append(q);ds.append(dt);ns.append(ok.sum())
 z=np.array(vals); print('H',h,'dates',len(z),'meanN',np.mean(ns) if ns else 0,'IC %.6f ICIR %.6f hit %.4f'%(np.mean(z),np.mean(z)/(np.std(z,ddof=1)+1e-12),np.mean(z>0)))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(np.array(ds)>=pd.Timestamp(lo+'-01-01'))&(np.array(ds)<=pd.Timestamp(hi+'-12-31'))]
  print(' ',lo+'-'+hi,len(q), 'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)) if len(q) else '')
ranks=sig.rank(axis=1,pct=True);print('turn10',ranks.diff(10).abs().mean(axis=1).dropna().mean(),'coverage',sig.notna().sum().sum()/sig.size)
