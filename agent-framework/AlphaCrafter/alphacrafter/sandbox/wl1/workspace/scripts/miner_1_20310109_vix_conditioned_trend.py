import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): D[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); v=v.set_index('date'); vc=[c for c in v.columns if c.lower() in ('close','value','price')][0]; v=v[vc].reindex(p.index).ffill()
r=p.pct_change(); vol=r.rolling(20).std(); base=p.shift(1).pct_change(10)/vol.shift(1)
z=(v.shift(1)-v.shift(1).rolling(252).median())/v.shift(1).rolling(252).std(); f=base.where(z<0,-base)
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); a=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 a=np.array(a); print('H',h,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('dates',len(p),'assets',len(D),'avg_names',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(f.shape[0]*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for yr in range(2020,2031):
 a=[]
 for dt in f.index[f.index.year==yr]:
  q=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 if len(a)>1:print('YEAR',yr,len(a),round(np.mean(a),5),round(np.mean(a)/np.std(a,ddof=1),4))
