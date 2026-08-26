import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-12-31'); P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); y=P.shift(-20)/P-1
# Volatility-shock continuation: medium momentum emphasized when recent volatility expands versus its baseline.
vol=r.rolling(20,min_periods=15).std(); shock=(r.rolling(5,min_periods=5).std()/vol).clip(0.25,4)
f=(P.pct_change(10)/(vol*np.sqrt(10))*shock).shift(1); f=f.sub(f.mean(axis=1),axis=0)
I=[]; N=[]; D=[]
for d in P.index:
 x=f.loc[d]; yy=y.loc[d]; ok=x.notna()&yy.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(yy[ok],method='spearman')
  if np.isfinite(z): I.append(z);N.append(ok.sum());D.append(d)
q=pd.Series(I,index=pd.DatetimeIndex(D)); print('dates',len(q),'avg_n',np.mean(N),'coverage',np.mean(N)/15,'IC20',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 w=q[(q.index.year>=a)&(q.index.year<=b)]; print('regime',a,b,'n',len(w),'ic',w.mean(),'icir',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True);b=f.iloc[i].rank(pct=True);ok=a.notna()&b.notna()
 if ok.sum()>=8:turn.append(np.abs(a[ok]-b[ok]).mean())
print('turnover',np.mean(turn))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290101_volshock_momentum_signal.csv',index=False)
