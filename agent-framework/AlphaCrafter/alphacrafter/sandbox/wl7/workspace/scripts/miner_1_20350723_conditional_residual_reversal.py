import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,4200)
   if d is not None and len(d): return d.set_index(pd.to_datetime(d.date))
  except Exception: pass
 return None
def main():
 ds={s:fetch(s) for s in U}; ds={s:d for s,d in ds.items() if d is not None}
 p=pd.DataFrame({s:d.close for s,d in ds.items()}).sort_index(); r=p.pct_change()
 # interpretable conditional residual reversal: reverse 10d relative to cross-sectional median,
 # risk-normalized, with stronger signal after broad market drawdowns.
 raw=r.rolling(10,min_periods=8).sum(); med=raw.median(axis=1); resid=raw.sub(med,axis=0)
 vol=r.rolling(30,min_periods=20).std()*np.sqrt(10)
 breadth=(raw<0).mean(axis=1)
 gate=(0.5+0.5*breadth).clip(.5,1.0)
 fac=(-resid/(vol+1e-12)).mul(gate,axis=0).shift(1)
 f=p.shift(-10).div(p)-1
 vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c): vals.append(c); ns.append(len(z))
 a=np.asarray(vals); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 for n in (252,756,1260):
  q=a[-n:]; print('recent',n,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
 # activation/top-quintile turnover and cross-sectional coverage
 top=fac.rank(axis=1,pct=True)>=.8; ch=[]
 for i in range(1,len(top)):
  z=top.iloc[i-1].notna()&top.iloc[i].notna()
  if z.sum()>=8: ch.append((top.iloc[i-1][z]!=top.iloc[i][z]).mean())
 print('coverage',fac.notna().mean().mean(),'turnover',np.mean(ch),'instruments',len(p.columns))
 fac.to_csv('scripts/miner_1_20350723_conditional_residual_reversal_signal.csv',index_label='date')
if __name__=='__main__': main()
