import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2034-11-22'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
px=pd.DataFrame(D).sort_index().ffill().loc[:cut]; r=np.log(px).diff()
# Downside-stability: lower downside semideviation, conditioned on nonnegative medium trend.
down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=20).mean())
trend=np.log(px/px.shift(60)); f=(-down)*(1+0.5*(trend>0)); f=f.rank(axis=1,pct=True); f=f.sub(f.median(axis=1),axis=0).shift(1)
for h in [5,10,20,40]:
 x=np.log(px.shift(-h)/px); vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(vals);print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],np.log(px.shift(-10)/px).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals);print('REG',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
f.to_csv('scripts/miner_2_20341124_downside_stability_signal.csv')
