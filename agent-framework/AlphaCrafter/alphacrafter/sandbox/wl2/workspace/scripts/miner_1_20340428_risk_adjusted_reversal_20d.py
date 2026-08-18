import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  z=d.copy(); z.date=pd.to_datetime(z.date); px[s]=z.set_index('date').close.astype(float)
P=pd.concat(px,axis=1).sort_index(); P=P.loc[P.index<=pd.Timestamp('2034-04-27')]
r=np.log(P/P.shift(1)); vol=r.rolling(20).std(); raw=r.rolling(20).sum()/(vol*np.sqrt(20)); F=-raw.shift(1)
# date-by-date cross-sectional IC, forward return from t close to t+h close
for h in [10,20,40]:
 fr=P.shift(-h)/P-1
 vals=[]; n=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); n.append(len(a))
 x=np.array(vals); print(h,'dates',len(x),'avg_n',round(np.mean(n),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round(np.mean(x>0),4))
# turnover and coverage (rank ordering change)
valid=F.notna().sum(axis=1)/15
ranks=F.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna()
print('coverage',round(valid.mean(),4),'turnover',round(turn.mean(),4),'period',P.index.min().date(),P.index.max().date(),'assets',len(px))
for label, sl in [('2020-2029',slice('2020','2029-12-31')),('2030-2034',slice('2030','2034-04-27'))]:
 vals=[]; fr=P.shift(-20)/P-1
 for dt in F.loc[sl].index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1]))
 x=np.array(vals); print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(),6))
