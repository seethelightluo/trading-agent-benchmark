import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None: fs[s]=d[['date','close']].assign(date=lambda x:pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
px=pd.DataFrame(fs).sort_index().ffill(); r=px.pct_change()
# Trend persistence: medium horizon return, penalized by downside risk and drawdown from 120d high.
ret=px/px.shift(60)-1; dv=r.where(r<0).rolling(40,min_periods=25).std(); dd=px/px.rolling(120,min_periods=80).max()-1
sig=(ret/(dv+1e-8) + 0.5*dd/(r.rolling(40,min_periods=25).std()+1e-8)).shift(1)
def eval(y):
 out=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 return pd.Series(dict(out)).dropna(),ns
fwd=px.shift(-10)/px-1; ic,ns=eval(fwd)
turn=[]
for dt in sig.index:
 x=sig.loc[dt].dropna().rank(pct=True); p=sig.shift(1).loc[dt].reindex(x.index).dropna().rank(pct=True)
 if len(p):turn.append(abs(x.reindex(p.index)-p).mean())
print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turn))
for h in [1,5,10,20]:
 q,_=eval(px.shift(-h)/px-1);print('decay',h,q.mean(),len(q))
for n in [365,750,1260]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350705_drawdown_trend_60d_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_1_20350705_drawdown_trend_60d_ic.csv',index=False)
