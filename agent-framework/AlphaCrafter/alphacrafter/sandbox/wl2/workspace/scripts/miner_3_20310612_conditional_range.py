import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); y=r.shift(-1)
# Range-location reversal only when cross-asset dispersion is elevated; risk scale by 20d vol.
look=60; hi=p.rolling(look,min_periods=30).max(); lo=p.rolling(look,min_periods=30).min(); loc=(p-lo)/(hi-lo)
vol=r.rolling(20,min_periods=10).std(); disp=r.std(axis=1).rolling(60,min_periods=30).mean(); gate=disp>disp.rolling(252,min_periods=100).median()
f=-(loc-.5)/(vol*np.sqrt(252)); f=f.where(gate, np.nan)
rows=[]
for i in range(len(f)-1):
 z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avgN',round(q.n.mean(),3),'active',round(gate.mean(),4),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA')
print('coverage_active %.4f turnover %.4f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.to_csv('scripts/miner_3_20310612_conditional_range_signal.csv')
