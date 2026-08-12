import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change(); b=r.mean(axis=1); rows={40:[],50:[],60:[]}
for t in range(150,len(p)-10):
 disp=r.iloc[t-20:t].std(axis=1).mean(); hist=[]
 # use trailing dispersion series
 for j in range(t-120,t): hist.append(r.iloc[j-20:j].std(axis=1).mean())
 for q in rows:
  if disp<=np.nanpercentile(hist,q):
   resid=(p.iloc[t]/p.iloc[t-40]-1)-(p.iloc[t]/p.iloc[t-40]-1).mean(); vol=r.iloc[t-40:t].std()*np.sqrt(40); f=-resid/vol
   z=pd.concat([f.rename('f'),(p.iloc[t+10]/p.iloc[t]-1).rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
   if len(z)>=8: rows[q].append((p.index[t],z.f.corr(z.r)))
for q,v in rows.items():
 o=pd.DataFrame(v,columns=['date','ic']).set_index('date'); recent=o.loc['2029-01-01':'2032-08-31']; print(q,'all',len(o),o.ic.mean(),o.ic.mean()/o.ic.std(), 'recent',len(recent),recent.ic.mean(),recent.ic.mean()/recent.ic.std(),(recent.ic>0).mean())
