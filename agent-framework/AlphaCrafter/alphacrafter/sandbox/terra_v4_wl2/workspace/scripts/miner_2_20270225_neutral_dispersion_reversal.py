import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); disp=r.sub(r.median(axis=1),axis=0).abs().mean(axis=1); base=disp.rolling(60,min_periods=30).median(); act=((disp/base)-1).clip(0,2)
r3=pd.DataFrame({a:p[a].pct_change(3) for a in A}); common=r3.median(axis=1); f=r3.sub(common,axis=0).mul(-act,axis=0)
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:f.at[dt,a] if dt in f.index else np.nan for a in A}; good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: sig.append((dt,a, vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  x=[];y=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); z=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(z) and i+h<len(p[a]): x.append(z);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(x)>=8: rows.append((dt,h,spearmanr(x,y).statistic,len(x)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_neutral_dispersion_reversal.csv',index=False)
w=out.pivot(index='date',columns='asset',values='signal');print('coverage',round(w.notna().mean().mean(),4),'rank_turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'artifact',len(out),'max_abs_library_correlation',None)
