import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=3400)
   if x is not None and len(x):
    y=x.set_index('date'); return y[['close','volume']].astype(float)
  except Exception: pass
 return None
def main():
 d={s:load(s) for s in SYMS}; d={s:x for s,x in d.items() if x is not None}; p=pd.concat({s:x.close for s,x in d.items()},axis=1).sort_index(); v=pd.concat({s:x.volume for s,x in d.items()},axis=1).reindex(p.index)
 # all inputs lagged one session: reversal is strengthened by unusually high contemporaneous (lagged) volume
 r=p.pct_change(20).shift(1); vs=(v/v.rolling(60,min_periods=30).median()).shift(1)
 fac=(-r*vs.clip(upper=4)).replace([np.inf,-np.inf],np.nan)
 for h in (5,10,20):
  fwd=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
  for dt in p.index:
   z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(z))
  ic=pd.Series(vals,index=ds).dropna(); ir=ic.mean()/ic.std(ddof=1)*np.sqrt(252)
  print('H',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic.mean(),6),'ICIR',round(ir,4),'hit',round((ic>0).mean(),4))
  if h==10: print('recent',[(n,round(ic.tail(n).mean(),6),round(ic.tail(n).mean()/ic.tail(n).std(ddof=1)*np.sqrt(252),4)) for n in (260,520,780) if len(ic)>=n])
 print('symbols',len(d),'volume_nonzero',round((v.replace(0,np.nan).notna().sum().sum()/v.size),4),'coverage',round((fac.notna().sum(axis=1)/len(d)).mean(),4),'cutoff',p.index.max())
 fac.tail(1).T.rename(columns={fac.index[-1]:'signal'}).to_csv('scripts/miner_2_20320805_volume_confirmed_reversal_20d_signal.csv')
if __name__=='__main__': main()
