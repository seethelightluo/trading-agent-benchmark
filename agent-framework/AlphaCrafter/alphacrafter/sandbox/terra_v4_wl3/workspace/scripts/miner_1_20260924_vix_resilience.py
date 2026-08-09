import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close
R=pd.DataFrame({s:ld('../persistent/stock_data/'+s+'.csv').pct_change() for s in U})
V=ld('../persistent/index_data/VIX.csv').pct_change()
# regime resilience: rolling asset avg return on days VIX rose minus return on VIX fell
F=pd.DataFrame({s:(R[s].where(V>0).rolling(40,min_periods=20).mean()-R[s].where(V<=0).rolling(40,min_periods=20).mean()) for s in U})
ics=[]; dates=[]; ns=[]
for i in range(len(R)-1):
 z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(R.index[i]);ns.append(len(z))
a=np.array(ics);print('dates',len(a),'avg names',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),'hit',np.mean(a>0),'coverage',F.notna().mean().mean())
for h in [5,10]:
 q=[]
 for i in range(len(R)-h):
  z=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(h,np.mean(q),np.mean(q)/np.std(q,ddof=1)*np.sqrt(len(q)),len(q))
for y in range(2020,2027):
 q=a[np.array([d.year for d in dates])==y];print(y,len(q),np.mean(q) if len(q) else np.nan)
print('turn',np.mean(F.rank(pct=True).diff().abs().mean(axis=1).dropna()))
