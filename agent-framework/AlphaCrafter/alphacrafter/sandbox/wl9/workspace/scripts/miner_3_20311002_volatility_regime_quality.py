import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(symbol=s,days=5000)
   if x is not None and len(x)>300:return x[['date','close']]
  except:pass

def main():
 d={s:f(s) for s in U};d={s:x for s,x in d.items() if x is not None}; c=pd.concat([x.set_index('date').close.rename(s) for s,x in d.items()],axis=1).sort_index().ffill(); r=np.log(c).diff()
 # Low realized volatility, conditioned on cross-asset stress: inverse 20d vol times positive cross-sectional dispersion.
 v=r.rolling(20).std(); disp=r.std(axis=1).rolling(20).mean(); stress=(disp/disp.rolling(252).median()).clip(0.5,2.0); sig=(-v).mul(stress,axis=0)
 for h in [5,10,20,40,60]:
  q=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],(c.shift(-h)/c-1).loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  q=pd.Series(q).dropna();print(h,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),(q>0).mean())
 q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],(c.shift(-20)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 q=pd.DataFrame(q,columns=['date','ic']).set_index('date')
 for n,a,b in [('2024-26','2024','2026'),('2027-29','2027','2029'),('2030','2030','2030'),('2031','2031','2031')]:
  x=q.loc[a:b].ic;print('regime',n,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252))
 print('assets',len(d),'dates',len(c),'coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
 sig.to_csv('scripts/miner_3_20311002_volatility_regime_quality_signal.csv',index_label='date')
if __name__=='__main__':main()
