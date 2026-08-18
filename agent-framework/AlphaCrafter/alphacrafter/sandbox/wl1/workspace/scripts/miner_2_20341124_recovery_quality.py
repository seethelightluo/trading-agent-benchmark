import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2034-11-22'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
px=pd.DataFrame(D).sort_index().ffill().loc[:cut]; r=np.log(px).diff()
lo60=px.rolling(60,min_periods=40).min(); recovery=np.log(px/lo60)
# downside semideviation with fixed-window zero fill, robustly defined across all assets
down2=(r.clip(upper=0)**2).rolling(30,min_periods=15).mean(); down=np.sqrt(down2)
f=(recovery/down).rank(axis=1,pct=True); f=f.sub(f.median(axis=1),axis=0).shift(1)
fr={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}
for h,x in fr.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals); print('REG',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
f.to_csv('scripts/miner_2_20341124_recovery_quality_signal.csv'); print('signal_path scripts/miner_2_20341124_recovery_quality_signal.csv')
