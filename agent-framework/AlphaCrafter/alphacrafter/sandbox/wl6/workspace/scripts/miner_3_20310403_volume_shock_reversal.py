import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.assign(date=pd.to_datetime(d.date)).set_index('date'); px[s]=d.close; vol[s]=d.volume.replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change()
# Volume-shock reversal: favor 20d losers, but only when the selloff occurred on
# unusually high aggregate volume; normalize by trailing risk for cross-asset comparability.
ret20=P.pct_change(20); risk=R.rolling(40,min_periods=25).std()
vbase=V.rolling(60,min_periods=30).mean(); vshock=(V.rolling(20,min_periods=10).mean()/vbase-1).clip(-2,2)
f=(-ret20/risk.replace(0,np.nan))*vshock
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=q.ic
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,6),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),8),'hit',round((a>0).mean(),6))
 print('years',q.groupby(q.index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),8))
