import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change()
# Candidate: residual multi-horizon momentum, scaled by idiosyncratic volatility.
# Remove common cross-asset movement at each date, then blend 20/60d residual returns.
raw20=P.pct_change(20); raw60=P.pct_change(60)
res20=raw20.sub(raw20.median(axis=1),axis=0)
res60=raw60.sub(raw60.median(axis=1),axis=0)
vol=R.rolling(30,min_periods=20).std()*np.sqrt(20)
F=((0.6*res20+0.4*res60)/vol).shift(1)
rows=[]
for h in [1,3,5,10,20,40]:
 y=P.shift(-h).div(P)-1; a=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: a.append((dt,z.f.corr(z.y),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(m,ir,(q.ic>0).mean()))
 for lo,hi in [(2020,2025),(2026,2029),(2030,2035)]:
  z=q.loc[str(lo):str(hi)].ic; print(' regime',lo,hi,'n',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA')
print('rows',len(P),'instruments',len(D),'coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.to_csv('../persistent/miner_1_20350608_residual_multihorizon_momentum_signal.csv')
