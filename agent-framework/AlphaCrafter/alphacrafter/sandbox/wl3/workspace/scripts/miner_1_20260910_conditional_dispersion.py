import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-09-09'
def load(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); return x.set_index('date')
def longf(x,name): return x.stack().rename_axis(['date','symbol']).rename(name).reset_index()
def panel():
 xs=[]
 for s in U:
  x=load(s); r=x.close.pct_change(); xs.append(pd.DataFrame({'date':x.index,'symbol':s,'r':r,'r5':x.close.pct_change(5),'vol':r.rolling(20,min_periods=15).std()}))
 a=pd.concat(xs).pivot(index='date',columns='symbol'); rr=a['r']; disp=rr.std(axis=1).rolling(20,min_periods=15).mean(); threshold=disp.rolling(60,min_periods=30).median()
 regime=np.where(disp>threshold,-1.0,1.0)
 sig=(-a['r5'].div(a['vol']*np.sqrt(20))).mul(regime,axis=0)
 out=longf(sig,'signal'); prices=pd.concat([load(s).close.rename(s) for s in U],axis=1); fwd=prices.shift(-1).div(prices)-1
 out=out.merge(longf(fwd,'fwd'),on=['date','symbol']).dropna(); out['regime']=out.date.map(pd.Series(regime,index=disp.index)); out['dispersion']=out.date.map(disp); return out
def ev(a):
 z=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1:
   c=spearmanr(g.signal,g.fwd).statistic
   if pd.notna(c): z.append((d,c)); ns.append(len(g))
 return pd.DataFrame(z,columns=['date','ic']).set_index('date'),ns
a=panel(); z,ns=ev(a); q=z.ic; rank=a.assign(rank=a.groupby('date').signal.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('conditional_dispersion_regime cutoff',cut,'dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',len(a)/(len(U)*a.date.nunique()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-09-09')]:
 v=z.loc[lo:hi].ic; print('regime_period',lo,hi,'n',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
prices=pd.concat([load(s).close.rename(s) for s in U],axis=1)
for h in [3,5,10]:
 f=prices.shift(-h).div(prices)-1; b=a[['date','symbol','signal']].merge(longf(f,'fwd'),on=['date','symbol']).dropna(); zz,nn=ev(b); print('decay',h,'n',len(zz),'IC',zz.ic.mean(),'ICIR',zz.ic.mean()/zz.ic.std(ddof=1))
a.to_csv('scripts/miner_1_20260910_conditional_dispersion_signal.csv',index=False)
