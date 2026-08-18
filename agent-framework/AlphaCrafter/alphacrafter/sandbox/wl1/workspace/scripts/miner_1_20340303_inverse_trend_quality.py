import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change(); down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(40)
f=(-np.log(px/px.shift(60))/(down+1e-8)).shift(1)
fr=px.pct_change(20).shift(-20); z=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
z=np.array(z);print('dates',len(z),'IC %.8f ICIR %.8f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)))
for label,mask in [('2020-23',pd.to_datetime(f.index).year<=2023),('2024-26',(pd.to_datetime(f.index).year>=2024)&(pd.to_datetime(f.index).year<=2026)),('2027-29',(pd.to_datetime(f.index).year>=2027)&(pd.to_datetime(f.index).year<=2029)),('2030-32',(pd.to_datetime(f.index).year>=2030)&(pd.to_datetime(f.index).year<=2032)),('2033+',pd.to_datetime(f.index).year>=2033)]:
 zz=[]
 for dt in f.index[mask]:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:zz.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 zz=np.array(zz);print(label,len(zz),np.nanmean(zz),np.nanmean(zz)/np.nanstd(zz,ddof=1))
out=f.copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_1_20340303_inverse_trend_quality_signal.csv',index=False)
