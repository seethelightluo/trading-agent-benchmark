import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A})
# Smoothed short-horizon reversal: equal-weight residual returns over 2d and 3d,
# then volatility-normalize. All inputs end at decision date.
r2=r.rolling(2,min_periods=2).sum(); r3=r.rolling(3,min_periods=3).sum()
res2=r2.sub(r2.median(axis=1),axis=0); res3=r3.sub(r3.median(axis=1),axis=0)
vol=r.rolling(20,min_periods=10).std(); raw=-(0.5*res2+0.5*res3).div(vol)
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:(raw.at[dt,a] if dt in raw.index else np.nan) for a in A}
 good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); z=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(z) and i+h<len(p[a]): f.append(z); y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270225_smoothed_residual_reversal23.csv',index=False)
wide=out.pivot(index='date',columns='asset',values='signal'); print('artifact',len(out),'coverage',round(out.signal.notna().mean(),4),'rank_turnover',round(wide.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
