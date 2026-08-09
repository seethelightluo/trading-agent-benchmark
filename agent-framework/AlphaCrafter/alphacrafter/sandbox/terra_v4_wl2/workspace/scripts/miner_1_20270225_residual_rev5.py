import pandas as pd,numpy as np
use=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in use:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index().loc[:'2027-02-25']; r=p.pct_change(); m=r.mean(axis=1)
b=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
# residual cumulative 5d return, contrarian; beta exposure removed using benchmark cumulative return
mr=m.rolling(5).sum(); ar=r.rolling(5).sum(); sig=-(ar-b.mul(mr,axis=0))
fwd=p.shift(-1).div(p)-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
print('dates',len(r),'avg_n',r.n.mean(),'coverage',(sig.notna().sum(axis=1)/15).mean())
for lab,sub in [('all',r),('2020-22',r.loc['2020':'2022']),('2023-24',r.loc['2023':'2024']),('2025-26',r.loc['2025':'2026']),('online',r.loc['2026-07-16':])]:
 if len(sub): print(lab,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1),(sub.ic>0).mean())
for h in [3,5,10]:
 y=p.shift(-h).div(p)-1; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna();print('h',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('turn',sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_residual_rev5.csv',index=False)
