import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change()
# Candidate: lagged risk-adjusted 30d trend, strengthened when cross-asset breadth confirms
mom=P.pct_change(30); vol=R.rolling(40,min_periods=25).std()*np.sqrt(30)
breadth=(R.rolling(20,min_periods=15).sum()>0).mean(axis=1)
# smooth confirmation avoids binary day noise; centered at neutral, positive breadth boosts trend
confirm=(breadth.rolling(10,min_periods=6).mean()-0.5).clip(-0.5,0.5)
F=(mom/vol).mul(1+confirm*1.5,axis=0).shift(1)
rows=[]
for h in [1,3,5,10]:
 y=P.shift(-h).div(P)-1; a=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: a.append((dt,z.f.corr(z.y),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(m,ir,(q.ic>0).mean()))
 for lo,hi in [(2020,2025),(2026,2029),(2030,2033)]:
  z=q.loc[str(lo):str(hi)].ic; print(' regime',lo,hi,'n',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA')
print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.to_csv('scripts/miner_1_20330805_breadth_confirmed_trend_signal.csv')
