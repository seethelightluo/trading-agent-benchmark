import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); r3=r.rolling(3).sum(); common=r3.median(axis=1); resid=r3.sub(common,axis=0)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].reindex(r.index).ffill()
# stress condition: VIX above trailing 60d median, using lagged VIX level
stress=(vix.shift(1)>vix.shift(1).rolling(60,min_periods=30).median())
raw=-resid
rows=[];sig=[]
for dt in r.index:
 if not stress.get(dt,False): continue
 vals=raw.loc[dt]; good=vals.dropna();
 if len(good)<8: continue
 med=good.median()
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if a not in vals or not np.isfinite(vals[a]) or dt not in p[a].index: continue
   i=p[a].index.get_loc(dt)
   if i+h>=len(p[a]): continue
   f.append(vals[a]-med); y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals.get(a,np.nan)) else np.nan))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_stress_residual_reversal.csv',index=False); print('coverage',out.signal.notna().mean(),'artifact',len(out))
if len(out): print('turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
