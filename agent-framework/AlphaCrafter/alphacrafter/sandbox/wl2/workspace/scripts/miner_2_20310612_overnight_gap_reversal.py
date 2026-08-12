import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: frames[s]=x.set_index('date')
# Overnight gap proxy: open-to-prior-close, reversed and normalized by 20d intraday volatility.
cl=pd.DataFrame({s:x.close.astype(float) for s,x in frames.items()}).sort_index().ffill()
op=pd.DataFrame({s:x.open.astype(float) for s,x in frames.items()}).reindex(cl.index).ffill()
r=cl.pct_change(); y=r.shift(-1); gap=op/cl.shift(1)-1; intr=(cl/op-1); vol=intr.rolling(20,min_periods=10).std(); f=-gap/(vol+1e-8)
rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((f.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('universe',len(frames),'dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a+'-'+b,'dates',len(z),'IC %.6f ICIR %.6f'%(len(z) and z.mean() or np.nan,len(z)>1 and z.mean()/z.std(ddof=1) or np.nan))
print('coverage %.4f turnover %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.to_csv('scripts/miner_2_20310612_overnight_gap_reversal_signal.csv')
