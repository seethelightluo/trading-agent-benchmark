import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); y=r.shift(-1)
vol=r.rolling(20,min_periods=10).std(); disp=r.sub(r.mean(axis=1),axis=0).abs().mean(axis=1).rolling(20,min_periods=10).mean(); base=disp.rolling(60,min_periods=40).median(); surprise=(disp/base-1).clip(lower=0); f=-(r.rolling(3,min_periods=3).sum()).div(vol*np.sqrt(252)).mul(surprise,axis=0)
rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((f.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.4f turnover %.4f active %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),(surprise>0).mean()))
f.to_csv('scripts/miner_3_20310515_dispersion_surprise_reversal_signal.csv')
