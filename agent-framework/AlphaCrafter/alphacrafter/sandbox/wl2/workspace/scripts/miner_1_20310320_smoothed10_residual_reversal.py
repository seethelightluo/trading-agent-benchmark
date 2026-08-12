import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.median(axis=1)
v=m.rolling(60,min_periods=40).var(); beta=r.rolling(60,min_periods=40).cov(m).div(v,axis=0)
res=r-beta.mul(m,axis=0)
f=-res.ewm(span=10,adjust=False,min_periods=10).mean()/r.rolling(20,min_periods=15).std()
def eval(y):
 rows=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
 a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
 return len(a),a.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),a
for h in [1,3,5,10]:
 y=p.shift(-h).div(p).sub(1)
 z=eval(y); print('h',h,'dates',z[0],'avgN',round(z[1],3),'IC',round(z[2],6),'ICIR',round(z[3],6),'hit',round(z[4],4),'coverage',round(z[5],4),'turn',round(z[6],4))
 if h==1:
  a=z[7]
  for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
   q=a.loc[mask].ic; print(nm,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
f.to_csv('scripts/miner_1_20310320_smoothed10_residual_signal.csv')
