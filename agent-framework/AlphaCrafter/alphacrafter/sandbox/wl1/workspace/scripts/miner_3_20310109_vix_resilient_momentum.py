import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
m=pd.read_csv('../persistent/index_data/VIX.csv'); m['date']=pd.to_datetime(m['date']); mc=pd.to_numeric(m.set_index('date').sort_index()['close'],errors='coerce')
prices={}
for s in U:
 try: d=get_index_daily_data(s,days=4100)
 except FileNotFoundError: d=get_stock_daily_data(s,days=4100)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); prices[s]=d.set_index('date')['close'].astype(float).sort_index()
p=pd.DataFrame(prices).sort_index(); r=np.log(p).diff(); ret20=p/p.shift(20)-1; vol20=r.rolling(20).std()*np.sqrt(20)
v=mc.reindex(p.index).ffill(); vr=np.log(v).diff(); beta=pd.DataFrame({s:r[s].rolling(60).cov(vr)/vr.rolling(60).var() for s in U},index=p.index)
raw=ret20/(vol20+1e-12)-.35*beta.mul(vr.rolling(5).mean(),axis=0); rank=raw.shift(1).rank(axis=1,pct=True); rows=[]; dates=[]; fwd=p.shift(-1)/p-1
for dt in rank.index:
 z=pd.concat([rank.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
ics=pd.Series(rows,index=dates).dropna(); print('dates',len(ics),'avg_names',p.notna().sum(axis=1).mean(),'daily IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),(ics>0).mean()))
for h in [5,10,20]:
 a=[]; yy=p.shift(-h)/p-1
 for dt in rank.index:
  z=pd.concat([rank.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna(); print('%dd IC %.6f ICIR %.6f'%(h,a.mean(),a.mean()/a.std(ddof=1)))
print('turnover',rank.diff().abs().mean(axis=1).dropna().mean())
for yr,g in ics.groupby(ics.index.year): print(yr,'%.5f n=%d'%(g.mean(),len(g)))
rank.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20310109_vix_resilient_momentum_signal.csv',index=False)
