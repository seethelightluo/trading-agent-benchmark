import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
P={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 P[s]=d[d.index<=end]
# align completed closes and OHLC
C=pd.DataFrame({s:d.close for s,d in P.items()}).sort_index().ffill(); O=pd.DataFrame({s:d.open for s,d in P.items()}).reindex(C.index).ffill(); H=pd.DataFrame({s:d.high for s,d in P.items()}).reindex(C.index).ffill(); L=pd.DataFrame({s:d.low for s,d in P.items()}).reindex(C.index).ffill()
rng=(H-L).replace(0,np.nan)
# bullish close location + open-close pressure; negative as prior experiment (higher factor predicts higher next return)
clv=2*(C-L)/rng-1; candle=(C-O)/rng
F=-(0.6*clv+0.4*candle).rolling(3).mean()
rets=C.pct_change();
def ics(h, idx=None):
 y=C.shift(-h)/C-1; out=[]; ns=[]
 for dt in F.index if idx is None else F.loc[idx].index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return np.array(out),np.array(ns)
for h in [1,5,10,20]:
 x,n=ics(h); print('H',h,'dates',len(x),'meanN',round(n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-07-15')]:
 x,n=ics(1,F.index[(F.index>=pd.Timestamp(lo))&(F.index<=pd.Timestamp(hi))]); print('REG',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
rank=F.rank(axis=1,pct=True); turn=(rank-rank.shift()).abs().mean(axis=1).dropna(); print('coverage',round(F.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(turn.mean(),6),'valid_dates',int(F.notna().sum(axis=1).ge(8).sum()))
# pooled date-level factor correlation with existing simple library proxies
mom=C/C.shift(20)-1; rev=-(C/C.shift(5)-1); peer=(C/C.shift(5)-1).apply(lambda row: row.map(lambda v: np.nan))
# leave-one-out median
rr=C.pct_change(5); peer=pd.DataFrame(index=C.index,columns=C.columns,dtype=float)
for dt,row in rr.iterrows():
 for s in C.columns: peer.loc[dt,s]=row.drop(labels=s).median()
for nm,X in [('mom20',mom),('rev5',rev),('peer5',peer)]:
 a=pd.concat([F.stack(),X.stack()],axis=1).dropna(); print('CORR',nm,round(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),6))
