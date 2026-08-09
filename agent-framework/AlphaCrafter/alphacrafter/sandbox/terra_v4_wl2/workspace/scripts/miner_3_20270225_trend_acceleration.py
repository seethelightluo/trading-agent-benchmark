import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in assets}
# Trend acceleration: recent 10d return relative to prior 30d return, risk normalized by 20d volatility.
r={a:p[a].pct_change() for a in assets}
raw={a:(p[a]/p[a].shift(10)-1 - (p[a].shift(10)/p[a].shift(40)-1))/((r[a].rolling(20).std()*np.sqrt(10))+1e-8) for a in assets}
idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]; sig=[]
for dt in idx:
 vals={a:raw[a].get(dt,np.nan) for a in assets}; good=[v for v in vals.values() if np.isfinite(v)]
 med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in assets: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  fac=[]; fwd=[]
  for a in assets:
   if dt not in p[a].index: continue
   ix=p[a].index.get_loc(dt); f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if ix+h<len(p[a]) and np.isfinite(f):
    y=p[a].iloc[ix+h]/p[a].iloc[ix]-1
    if np.isfinite(y): fac.append(f);fwd.append(y)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fwd).statistic,len(fac)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_trend_acceleration.csv',index=False)
wide=out.pivot(index='date',columns='asset',values='signal'); print('turnover',round(wide.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
