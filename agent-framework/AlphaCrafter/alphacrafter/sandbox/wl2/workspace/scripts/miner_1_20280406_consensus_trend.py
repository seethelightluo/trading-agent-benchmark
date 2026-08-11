import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d): D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Consensus trend: agreement of 10/30/60d returns, scaled by 30d volatility; lagged one bar.
r10=r.rolling(10,min_periods=10).sum(); r30=r.rolling(30,min_periods=30).sum(); r60=r.rolling(60,min_periods=60).sum(); v=r.rolling(30,min_periods=30).std()*np.sqrt(30)
# smooth bounded agreement rewards consistent sign and penalizes disagreement
agree=(np.sign(r10)+np.sign(r30)+np.sign(r60))/3
f=((0.35*r10+0.40*r30+0.25*r60)/v)*((1+agree)/2).clip(.25,1.0)
f=f.shift(1)
def ev(y):
 a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for h in [1,3,5,10]: print('h',h,'dates %d avgN %.2f IC %.6f ICIR %.6f hit %.4f'%ev(np.log(p).shift(-h)-np.log(p)))
print('coverage %.4f turnover %.6f dates %d instruments %d period %s %s'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),len(p),len(D),p.index.min(),p.index.max()))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:
 y=np.log(p).shift(-1)-np.log(p); a=[]
 for dt in f.loc[s].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q)
 print(n,'dates',len(a),'IC %.6f ICIR %.6f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1)))
