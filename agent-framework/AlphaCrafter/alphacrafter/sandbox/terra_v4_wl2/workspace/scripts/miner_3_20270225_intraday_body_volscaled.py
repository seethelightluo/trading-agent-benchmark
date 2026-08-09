import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
r={a:d[a].close.pct_change() for a in A}; vol={a:r[a].rolling(20,min_periods=10).std() for a in A}
raw={a:-(d[a].close-d[a].open)/(d[a].high-d[a].low).replace(0,np.nan)/vol[a] for a in A}
rows=[];sig=[]
for dt in sorted(set().union(*[set(x.index) for x in d.values()])):
 v={a:raw[a].get(dt,np.nan) for a in A}; good=[x for x in v.values() if np.isfinite(x)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A:sig.append((dt,a,v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan))
 f=[];y=[]
 for a in A:
  if dt not in d[a].index:continue
  i=d[a].index.get_loc(dt);z=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
  if np.isfinite(z) and i+1<len(d[a]):f.append(z);y.append(d[a].close.iloc[i+1]/d[a].close.iloc[i]-1)
 if len(f)>=8:rows.append((dt,spearmanr(f,y).statistic,len(f)))
x=pd.DataFrame(rows,columns=['date','ic','n']);print('dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=x.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_intraday_body_volscaled.csv',index=False);print('turnover',round(out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
