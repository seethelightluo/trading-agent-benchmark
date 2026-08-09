import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}; P=pd.DataFrame(p).sort_index(); r=P.pct_change()
# Asset-specific beta-neutral relative momentum: 20d asset return minus contemporaneous median, contrasted with 5d.
med=r.median(axis=1); ex=r.sub(med,axis=0)
raw=ex.rolling(20,min_periods=15).sum()-ex.rolling(5,min_periods=5).sum()
# Invert recent overshoot: medium trend with short-term exhaustion, a simple interpretable hybrid.
rows=[]; sig=[]
for dt in P.index:
 v=raw.loc[dt]; c=v.dropna().median() if v.notna().sum()>=8 else np.nan
 for a in A:sig.append((dt,a,v[a]-c if np.isfinite(c) and np.isfinite(v[a]) else np.nan))
 for h in [1,5,10]:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1;q=pd.concat([(v-c).rename('f'),y.rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:rows.append((dt,h,spearmanr(q.f,q.y).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=d[d.h==h];print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  x=q.set_index('date').loc[lo:hi].ic;print(lo,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_exhaustion.csv',index=False);w=out.pivot(index='date',columns='asset',values='signal');print('coverage',out.signal.notna().mean(),'turnover',w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
