import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15'); D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); D[s]=x.set_index('date')['close']
 except Exception as e: print('missing',s,e)
prices=pd.DataFrame(D).sort_index(); R=prices.pct_change(); bench=R.median(axis=1); resid=R.sub(bench,axis=0); Y=R.shift(-1)
for w in [2,3,5,10]:
 F=-resid.rolling(w,min_periods=w).sum(); rows=[]
 for dt in F.index:
  g=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: rows.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); ic=a.ic.to_numpy(); ranks=F.rank(axis=1,pct=True); turn=ranks.diff().abs().mean().mean()
 print('w',w,'dates',len(a),'avg_names',a.n.mean(),'coverage',F.notna().sum().sum()/F.size,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',turn)
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)].ic; print(' regime',lo,hi,'dates',len(q),'ICIR',q.mean()/q.std(ddof=1))
