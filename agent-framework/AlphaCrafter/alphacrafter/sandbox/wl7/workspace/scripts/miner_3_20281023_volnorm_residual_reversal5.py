import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-10-22')
P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change()
# Volatility-normalized 5d reversal, centered against contemporaneous cross-sectional mean.
# Divide trailing 5d return by trailing 20d daily volatility; use negative score.
vol=r.rolling(20,min_periods=15).std()
raw=-(P.pct_change(5)/ (vol*np.sqrt(5))).replace([np.inf,-np.inf],np.nan)
# residualize cross-sectionally so signal measures relative reversal, not market direction
f=raw.sub(raw.mean(axis=1),axis=0)
rows=[]; ns=[]; turns=[]
for i in range(len(P)-1):
 x=f.iloc[i]; y=r.iloc[i+1]; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(z): rows.append((P.index[i],z)); ns.append(ok.sum())
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turns.append(np.abs(a[ok]-b[ok]).mean())
q=pd.DataFrame(rows,columns=['date','ic']).set_index('date'); ic=q.ic
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15)
print('IC',ic.mean(),'ICIR_daily',ic.mean()/ic.std(ddof=1),'ICIR_annualized',ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turns))
for h in [1,5,10]:
 a=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   z=x[ok].corr(y[ok],method='spearman')
   if np.isfinite(z): a.append(z)
 a=np.array(a); print('horizon',h,'dates',len(a),'IC',a.mean(),'ICIR_daily',a.mean()/a.std(ddof=1))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 z=ic[(ic.index.year>=a)&(ic.index.year<=b)]; print('regime',a,b,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20281023_volnorm_residual_reversal5_signal.csv',index=False)
