import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date')
# daily close-location value (CLV) accumulated over 10 sessions, blended with 20d trend
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index().ffill()
H=pd.DataFrame({s:x.high.astype(float) for s,x in D.items()}).reindex(C.index).ffill(); L=pd.DataFrame({s:x.low.astype(float) for s,x in D.items()}).reindex(C.index).ffill()
clv=((2*C-H-L)/(H-L).replace(0,np.nan)).rolling(10,min_periods=7).mean()
r=C.pct_change(); trend=r.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()
f=(trend/(vol*np.sqrt(20)))*clv
y=r.shift(-1); rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((f.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avgN',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.4f turnover %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.to_csv('scripts/miner_3_20310724_clv_signal.csv')
