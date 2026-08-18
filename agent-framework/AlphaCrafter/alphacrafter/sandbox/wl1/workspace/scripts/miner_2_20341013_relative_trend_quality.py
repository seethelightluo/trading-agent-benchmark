import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2034-10-13'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
px=pd.DataFrame(D).sort_index().ffill().loc[:cut]; r=np.log(px).diff()
# Relative trend quality: 20d momentum versus cross-sectional median, scaled by downside risk; stronger when breadth confirms
mom=r.rolling(20,min_periods=15).sum(); vol=r.rolling(30,min_periods=20).std(); down=r.clip(upper=0).rolling(30,min_periods=20).std()
bread=(mom>0).mean(axis=1); gate=(0.5+1.0*(bread>0.60)+0.5*(bread<0.40))
f=((mom/vol) - (mom/vol).median(axis=1).values[:,None]) * gate.values[:,None]
f=f.shift(1)
fr={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}
for h,x in fr.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals); print('H',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
rank=f.rank(axis=1,pct=True);print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals);print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('signal_path', 'scripts/miner_2_20341013_relative_trend_quality_signal.csv')
f.to_csv('scripts/miner_2_20341013_relative_trend_quality_signal.csv')
