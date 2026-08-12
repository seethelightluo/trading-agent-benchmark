import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=5000)
 return d
D={s:fetch(s) for s in U}; c=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
r=np.log(c).diff(); mom=r.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()
defensive=mom[['XAU','US10Y','CN10Y']].mean(axis=1)-mom[['SPX','NDX','000300.SH']].mean(axis=1)
lead=defensive.rolling(5,min_periods=3).mean()
base=mom.div(vol.replace(0,np.nan)); f=base.sub(base.median(axis=1),axis=0).mul(1+lead.clip(-.5,.5),axis=0).shift(1)
for h in [5,10,20,30]:
 fr=c.shift(-h)/c-1;a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z))
 a=np.asarray(a);print(h,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6),round(np.mean(a>0),4),round(np.mean(ns),2))
fr=c.shift(-10)/c-1;a=[]
for dt in f.index[f.index>='2027-01-01']:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
a=np.asarray(a);print('recent',len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
print('dates',c.index.min(),c.index.max(),'median assets',c.notna().sum(axis=1).median())
f.rename_axis('date').to_csv('scripts/miner_1_20281214_defensive_lead_riskadj_signal.csv')
