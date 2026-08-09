import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date'); D[s]=d
cl=pd.DataFrame({s:d['close'] for s,d in D.items()}).sort_index(); re=cl.pct_change()
# trend consistency: fraction positive daily returns, signed by 20d return and volatility-adjusted
cons=(re.rolling(20).mean()/re.rolling(20).std()).shift(1)
# alternative: signed fraction, excess over market cross-sectional median
f=cons.sub(cons.median(axis=1),axis=0)
print('instruments',len(D),'dates',cl.index.min(),cl.index.max())
for h in [1,5,10]:
 fr=cl.shift(-h)/cl-1
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',np.nanmean([pd.concat([f.loc[x],fr.loc[x]],axis=1).dropna().shape[0] for x in f.index if len(pd.concat([f.loc[x],fr.loc[x]],axis=1).dropna())>=8]),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for period in [("2020","2022"),("2023","2024"),("2025","2026")]:
 a=[]
 for dt in f.loc[period[0]:period[1]].index:
  z=pd.concat([f.loc[dt],(cl.shift(-1)/cl-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a); print(period,len(a),a.mean(),a.mean()/a.std())
# save artifact
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20261217_consistency_signal.csv')
