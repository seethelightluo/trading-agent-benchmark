import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2028-12-17'); P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); m=r.mean(axis=1)
# residual return versus rolling 60d beta to equal-weight cross-asset benchmark
cov=r.rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
res5=r.rolling(5,min_periods=5).sum().sub(beta.mul(m.rolling(5,min_periods=5).sum(),axis=0))
vol=r.rolling(30,min_periods=20).std(); f=-res5/(vol*np.sqrt(5)); f=f.sub(f.mean(axis=1),axis=0)
ics=[]; ns=[]; turns=[]
for i in range(len(P)-20):
 x=f.iloc[i]; y=P.iloc[i+20]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(z): ics.append((P.index[i],z)); ns.append(ok.sum())
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turns.append(np.abs(a[ok]-b[ok]).mean())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
print('dates',len(q),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC20',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turn',np.mean(turns))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 w=z[(z.index.year>=a)&(z.index.year<=b)]; print('regime',a,b,'n',len(w),'ic',w.mean(),'icir',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281218_residual_reversal5_signal.csv',index=False)
