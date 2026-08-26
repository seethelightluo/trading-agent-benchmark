import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(symbol=s,days=5000)
   if x is not None and len(x)>300:return x[['date','close']]
  except:pass
p={s:get(s) for s in U};p={s:x for s,x in p.items() if x is not None}
c=pd.concat([x.set_index('date').close.rename(s) for s,x in p.items()],axis=1).sort_index().ffill(); r=c.pct_change(); r60=c/c.shift(60)-1
down=r.clip(upper=0).pow(2).rolling(60,min_periods=30).mean().pow(.5)*np.sqrt(252)
sig=(-r60/down.replace(0,np.nan)).shift(1)
for h in [10,20,40,60]:
 f=c.shift(-h)/c-1;qs=[];ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d].rename('s'),f.loc[d].rename('f')],axis=1).dropna()
  if len(z)>=8:
   q=z.corr(method='spearman').iloc[0,1]
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);print(h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('assets',len(p),'dates',len(c),'coverage',round(sig.notna().mean().mean(),4));rank=sig.rank(axis=1,pct=True);print('turnover_proxy',round(rank.diff().abs().mean(axis=1).dropna().mean(),6));sig.to_csv('scripts/miner_2_20330106_downside_reversal_signal.csv',index_label='date')
