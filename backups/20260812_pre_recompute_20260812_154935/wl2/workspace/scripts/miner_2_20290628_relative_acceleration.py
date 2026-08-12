import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close
P=pd.concat(D,axis=1).sort_index().ffill(); R=P.pct_change()
# Relative acceleration: asset 10d relative return versus universe median, minus 40d relative return; risk scaled.
csmed=R.median(axis=1)
rel10=P.pct_change(10).sub(P.pct_change(10).median(axis=1),axis=0)
rel40=P.pct_change(40).sub(P.pct_change(40).median(axis=1),axis=0)
vol=R.rolling(20).std()
F=((rel10-rel40/4)/vol).shift(1)
ics={}; ns={}; turns=[]; prev=None
for h in [1,3,5,10]:
 vals=[]; nn=[]
 fr=P.pct_change(h).shift(-h)
 for i in range(45,len(P)-h):
  z=pd.concat([F.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nn.append(len(z))
 x=np.asarray(vals); ic=x.mean(); ir=ic/x.std(ddof=1)
 print('h',h,'dates',len(x),'avgN',round(np.mean(nn),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(x>0),4))
 if h==1:
  for i in range(45,len(P)):
   q=F.iloc[i].rank(pct=True)
   if q.notna().sum()>=8:
    if prev is not None: turns.append((q-prev).abs().mean())
    prev=q
  print('turnover',round(float(np.nanmean(turns)),6),'coverage',round(float(F.notna().sum().sum()/(F.shape[0]*F.shape[1])),6))
for label,lo in [('2020-2022','2020-01-01'),('2023-2025','2023-01-01'),('2026-2027','2026-01-01'),('2028+','2028-01-01')]:
 vals=[]; fr=P.pct_change().shift(-1)
 for i in range(45,len(P)-1):
  if P.index[i]<pd.Timestamp(lo): continue
  z=pd.concat([F.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals);print(label,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
