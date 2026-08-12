import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=4100)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); y=r.shift(-1); m=r.mean(axis=1); res=r.sub(m,axis=0); rv=res.rolling(20,min_periods=10).std(); raw=-(res.rolling(3,min_periods=3).sum()/rv)
v=get_index_daily_data('VIX',days=4100).set_index('date').close.astype(float).reindex(p.index).ffill(); gate=v>v.rolling(252,min_periods=100).quantile(.7); f=raw.where(gate,np.nan)
rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:rows.append((p.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'active',gate.mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20310710_vix_residual_reversal_3d_signal.csv')
