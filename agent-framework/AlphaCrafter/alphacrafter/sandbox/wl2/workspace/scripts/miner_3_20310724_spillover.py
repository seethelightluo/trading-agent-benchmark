import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4100)
    if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
    if x is not None: D[s]=x.set_index('date').close.astype(float)
p_=pd.DataFrame(D).sort_index().ffill(); r=p_.pct_change()
# Volatility-scaled cross-asset spillover: at each date, each asset's score is
# the average 5d return of all other assets, standardized by their trailing 20d vol.
ret5=r.rolling(5,min_periods=5).sum(); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
z=ret5/(vol20.replace(0,np.nan)); f=pd.DataFrame(index=z.index,columns=z.columns,dtype=float)
for s in z.columns: f[s]=z.drop(columns=s).mean(axis=1)
y=r.shift(-1); rows=[]
for i in range(len(f)-1):
    q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: rows.append((f.index[i],q.f.corr(q.y),len(q)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avgN',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 x=q.loc[a:b].ic; print(a,len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
print('coverage %.4f turnover %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.to_csv('scripts/miner_3_20310724_spillover_signal.csv')
