import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().ffill(); r=P.pct_change(); ret=P/P.shift(20)-1
# penalize downside only, rewarding persistent gains; interpretable tail-risk adjusted momentum
neg=r.where(r<0,0).rolling(40).std()*np.sqrt(252)
f=ret/neg.replace(0,np.nan)
fw=P.shift(-10)/P-1
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(f.loc[dt][ok],y.loc[dt][ok]).statistic); ns.append(ok.sum())
 a=np.asarray(vals); print('horizon',h,'dates',len(a),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC %.8f ICIR %.8f hit %.5f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
for name,mask in [('2020-2024',P.index.year<=2024),('2025-2027',(P.index.year>=2025)&(P.index.year<=2027)),('2028-2029',P.index.year>=2028)]:
 vals=[]
 for dt in f.index[mask]:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt][ok],fw.loc[dt][ok]).statistic)
 a=np.array(vals); print(name,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
