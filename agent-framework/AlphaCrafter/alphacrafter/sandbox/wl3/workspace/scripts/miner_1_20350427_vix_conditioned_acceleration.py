import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=6000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float).sort_index()
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vs=vix.set_index('date').close.astype(float).reindex(px.index).ffill()
base=(px/px.shift(20)-1-(px/px.shift(60)-1)/3)/(r.rolling(20).std()*np.sqrt(252)); stress=(vs>vs.rolling(60,min_periods=30).median()).astype(float); f=base.mul(1-0.75*stress,axis=0)
def calc(h):
 y=px.shift(-h)/px-1; rows=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('span',px.index.min().date(),px.index.max().date(),'dates/assets',len(px),len(D))
for h in [1,5,10,20]:
 a=calc(h); print('horizon',h,'dates',len(a),'avg_n',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 if h==10:
  print('recent252',a.tail(252).ic.mean(),a.tail(252).ic.mean()/a.tail(252).ic.std(ddof=1)); ranks=f.rank(axis=1,pct=True);print('turnover',((ranks-ranks.shift()).abs().mean(axis=1)).dropna().mean())
  for i,ix in enumerate(np.array_split(np.arange(len(a)),4),1):
   b=a.iloc[ix];print('block',i,len(b),b.ic.mean(),b.ic.mean()/b.ic.std(ddof=1))
f.to_csv('factors/miner_1_20350427_vix_conditioned_acceleration_10d_signal.csv',index_label='date')
print('signal_artifact',f.shape)
