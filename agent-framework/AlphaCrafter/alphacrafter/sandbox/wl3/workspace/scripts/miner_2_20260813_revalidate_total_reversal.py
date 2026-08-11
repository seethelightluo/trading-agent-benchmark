import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-08-12')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
C=pd.concat({s:x.close for s,x in D.items()},axis=1).sort_index(); O=pd.concat({s:x.open for s,x in D.items()},axis=1).reindex(C.index).ffill(); R=C.pct_change(); V=R.rolling(20,min_periods=10).std(); F=-(O/C.shift(1)-1 + C/O-1)/V
out=[]
for dt in F.index:
 q=pd.DataFrame({'f':F.loc[dt],'y':R.shift(-1).loc[dt]}).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: out.append((dt,spearmanr(q.f,q.y).statistic,len(q)))
Z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');
print('cutoff',cut.date(),'dates',len(Z),'avg_n',Z.n.mean(),'IC',Z.ic.mean(),'ICIR',Z.ic.mean()/Z.ic.std(ddof=1),'hit',(Z.ic>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15'),('2026-07-16','2026-08-12')]:
 z=Z.loc[lo:hi].ic;print('regime',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [5,10]:
 q=[]; Y=C.pct_change(h).shift(-h)
 for dt in F.index:
  a=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1:q.append(spearmanr(a.f,a.y).statistic)
 print('decay',h,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
rank=F.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',F.notna().sum().sum()/(F.shape[0]*F.shape[1]))
