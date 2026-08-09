import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={};F={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();p=d.close; r=p.pct_change();P[a]=p
 # downside deviation proxy: mean negative-return magnitude, positive days contribute zero
 F[a]=-(-r.clip(upper=0)).rolling(20,min_periods=15).mean()
common=sorted(set().union(*[set(p.index) for p in P.values()]))
def go(h):
 out=[]
 for dt in common:
  v=[];y=[]
  for a in A:
   f=F[a].get(dt,np.nan)
   if dt not in P[a].index or not np.isfinite(f):continue
   i=P[a].index.get_loc(dt)
   if i+h>=len(P[a]):continue
   z=P[a].iloc[i+h]/P[a].iloc[i]-1
   if np.isfinite(z):v.append(f);y.append(z)
  if len(v)>=8:out.append((dt,spearmanr(v,y).statistic,len(v)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
x=go(1);print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 z=x.loc[lo:hi].ic; print(lo,len(z),z.mean(),z.mean()/z.std())
for h in [3,5,10]:
 z=go(h);print('h',h,len(z),z.ic.mean(),z.ic.mean()/z.ic.std())
out=[pd.DataFrame({'date':F[a].index,'asset':a,'signal':F[a].values}) for a in A];pd.concat(out).to_csv('../persistent/factor_signals_miner_3_20270114_downside.csv',index=False)
