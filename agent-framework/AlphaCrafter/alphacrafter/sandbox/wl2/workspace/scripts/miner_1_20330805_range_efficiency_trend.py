import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change()
# Range-efficiency trend: directional displacement divided by realized path length,
# multiplied by volatility-normalized magnitude; lagged to avoid look-ahead.
net=R.rolling(30,min_periods=24).sum(); path=R.abs().rolling(30,min_periods=24).sum()
vol=R.rolling(60,min_periods=40).std()*np.sqrt(30)
F=((net/path)*(net/vol)).shift(1)
for h in [1,3,5,10]:
 y=P.shift(-h).div(P)-1; out=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:out.append((dt,z.f.corr(z.y),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(m,ir,(q.ic>0).mean()))
 for lo,hi in [(2020,2025),(2026,2029),(2030,2033)]:
  a=q.loc[str(lo):str(hi)].ic; print(' regime',lo,hi,'n',len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)) if len(a)>1 else 'NA')
print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.to_csv('scripts/miner_1_20330805_range_efficiency_trend_signal.csv')
