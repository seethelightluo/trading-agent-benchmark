import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=3400)
   if x is not None and len(x): return x.set_index('date').close.astype(float)
  except Exception: pass
 return None
def main():
 d={s:load(s) for s in SYMS}; d={s:x for s,x in d.items() if x is not None}
 p=pd.DataFrame(d).sort_index(); r=p.pct_change()
 # Lagged short-horizon reversal, scaled by idiosyncratic volatility and market-wide
 # cross-sectional dispersion. Dispersion gate makes reversal stronger only when
 # opportunities are differentiated; every input is shifted one completed session.
 ret10=p.pct_change(10).shift(1); v20=r.rolling(20,min_periods=15).std().shift(1)
 csdisp=ret10.std(axis=1).where(ret10.notna().sum(axis=1)>=8).shift(1)
 baseline=csdisp.rolling(120,min_periods=60).median().shift(1)
 gate=(csdisp/(baseline+1e-12)).clip(.5,2.0)
 fac=(-ret10/(v20+1e-12)).mul(gate, axis=0).replace([np.inf,-np.inf],np.nan)
 for h in (5,10,20):
  fw=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
  for dt in p.index:
   z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(z))
  ic=pd.Series(vals,index=ds).dropna(); ir=ic.mean()/ic.std(ddof=1)*np.sqrt(252)
  print('H',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic.mean(),6),'ICIR',round(ir,4),'hit',round((ic>0).mean(),4))
  if h==10: print('recent',[(n,round(ic.tail(n).mean(),6),round(ic.tail(n).mean()/ic.tail(n).std(ddof=1)*np.sqrt(252),4)) for n in (260,520,780) if len(ic)>=n])
 print('symbols',len(d),'coverage',round((fac.notna().sum(axis=1)/len(d)).mean(),4),'turnover',round((fac.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).dropna().mean(),4),'cutoff',p.index.max())
 fac.tail(1).T.rename(columns={fac.index[-1]:'signal'}).to_csv('scripts/miner_2_20321014_dispersion_gated_reversal_signal.csv')
if __name__=='__main__': main()
