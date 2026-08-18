import pandas as pd, numpy as np, glob
px={}
for f0 in glob.glob('../persistent/stock_data/*.csv'):
 d=pd.read_csv(f0);d.date=pd.to_datetime(d.date);px[f0.split('/')[-1][:-4]]=d.set_index('date').close
P=pd.DataFrame(px).sort_index().loc[:'2026-07-15'];r=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);vv=v.set_index('date').close.reindex(P.index).ffill();vz=((vv-vv.rolling(60).mean())/vv.rolling(60).std()).clip(lower=0).fillna(0)
f=-r.rolling(5).sum().multiply((1+vz).values,axis=0)
for h in [1,5,10]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+h]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(a);print('h',h,'dates',len(a),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
for lab,lo,hi in [('2020-22','2020-01-01','2023-01-01'),('2023-24','2023-01-01','2025-01-01'),('2025-26','2025-01-01','2026-07-16')]:
 a=[]
 for i,dt in enumerate(P.index[:-1]):
  if pd.Timestamp(lo)<=dt<pd.Timestamp(hi):
   z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.asarray(a);print(lab,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
print('range',P.index.min(),P.index.max())
