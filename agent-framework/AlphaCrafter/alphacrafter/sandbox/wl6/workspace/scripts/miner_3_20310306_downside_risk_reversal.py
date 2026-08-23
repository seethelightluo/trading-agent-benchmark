import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); ret=P.pct_change(20); tot=R.rolling(40,min_periods=25).std(); down=R.where(R<0).rolling(40,min_periods=5).std().fillna(tot)
f=-ret/(tot+down+1e-12)
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); q.date=pd.to_datetime(q.date); q=q.set_index('date'); a=q.ic
 print('H',h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean())
 print('years',q.groupby(q.index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
