import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}; sig={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); p=d.close
 prices[a]=p; r=p.pct_change(); sig[a]=-(r.rolling(5,min_periods=5).std()/r.rolling(20,min_periods=15).std())
common=sorted(set().union(*[set(x.index) for x in prices.values()]))
def calc(h):
 rows=[]
 for dt in common:
  v=[]; y=[]
  for a in assets:
   if dt not in sig[a].index: continue
   f=sig[a].loc[dt]; ix=prices[a].index.get_indexer([dt])[0] if dt in prices[a].index else -1
   if ix<0 or ix+h>=len(prices[a]) or not np.isfinite(f): continue
   z=prices[a].iloc[ix+h]/prices[a].iloc[ix]-1
   if np.isfinite(z):v.append(f);y.append(z)
  if len(v)>=8: rows.append((dt,spearmanr(v,y).statistic,len(v)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
x=calc(1);print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 z=x.loc[lo:hi].ic; print(lo,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()) if len(z) else '')
for h in [3,5,10]:
 z=calc(h);print('h',h,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std()))
out=[]
for a in assets: out.append(pd.DataFrame({'date':sig[a].index,'asset':a,'signal':sig[a].values}))
pd.concat(out).to_csv('../persistent/factor_signals_miner_3_20270114_vol_compression.csv',index=False)
