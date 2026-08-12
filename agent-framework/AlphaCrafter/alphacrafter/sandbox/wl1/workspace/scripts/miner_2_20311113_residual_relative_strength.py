import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 if x is not None and len(x)>100:
  x=x[['date','close']].copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Residual relative strength: asset 20d return less contemporaneous equal-weight benchmark return,
# scaled by idiosyncratic 30d volatility; all inputs end at t and signal is lagged.
bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
res20=resid.rolling(20,min_periods=15).sum(); iv=resid.rolling(30,min_periods=20).std()
f=(res20/(iv+1e-8)).shift(1)
print('candidate residual_relative_strength_20d')
for h in [1,5,10,20]:
 q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('horizon',h,'dates',len(q),'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
# regime split by benchmark trailing 60d sign
q=[]; reg=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],np.log(p.iloc[i+10]/p.iloc[i])],axis=1).dropna()
 if len(z)>=8:
  q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));reg.append('up' if bench.iloc[max(0,i-59):i+1].sum()>0 else 'down')
q=pd.Series(q);reg=pd.Series(reg,index=q.index)
for x in ['up','down']:
 a=q[reg==x];print('regime',x,'dates',len(a),'ic',round(a.mean(),6),'icir',round(a.mean()/a.std(ddof=1),6))
print('dates',len(p),'assets',len(D),'coverage',round(f.notna().mean().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20311113_residual_relative_strength_signal.csv',index=False)
