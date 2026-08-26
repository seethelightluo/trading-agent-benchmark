import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-04-26')
def f(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
p=pd.concat({a:f(a) for a in A},axis=1).sort_index().loc[:E].ffill(); r=p.pct_change(); b=r.mean(axis=1); rr=r.sub(b,axis=0)
# residual 5d reversal, risk scaled and lagged
s=(-(1+rr).rolling(5).apply(np.prod,raw=True)+1)/(rr.rolling(20).std()+.01); s=s.shift(1)
rows=[]
for t in p.index:
 x=s.loc[t]; y=p.shift(-10).loc[t]/p.loc[t]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((t,z,int(ok.sum())))
z=np.array([x[1] for x in rows]); print('period',rows[0][0].date(),rows[-1][0].date(),'dates',len(z),'avgN',np.mean([x[2] for x in rows])); print('IC10',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'annualized',z.mean()/z.std(ddof=1)*np.sqrt(252),'hit',np.mean(z>0),'coverage',np.mean([x[2]/15 for x in rows])); print('turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex([x[0] for x in rows]).mean())
for h in [1,5,10,20]:
 q=[]; y=p.shift(-h)/p-1
 for t in p.index:
  x=s.loc[t]; yy=y.loc[t]; ok=x.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],yy[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n in [365,750,1260]:
 q=z[-n:];print('recent',n,q.mean(),q.mean()/q.std(ddof=1))
o=pd.DataFrame({'date':s.index}); [o.__setitem__(a,s[a].values) for a in A];o.to_csv('factors/miner_1_20350426_residual_short_reversal_10d_signal.csv',index=False);pd.DataFrame(rows,columns=['date','ic','n']).to_csv('factors/miner_1_20350426_residual_short_reversal_10d_ic.csv',index=False)
