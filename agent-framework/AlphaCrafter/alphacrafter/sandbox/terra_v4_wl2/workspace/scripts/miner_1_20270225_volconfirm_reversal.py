import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-25')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# Elevated-volume short-term reversal: reverse prior 3-day return, scaled by log volume surprise.
F={a: -(D[a].close.pct_change(3))*np.log1p(D[a].volume/(D[a].volume.rolling(20).median()+1e-12)) for a in A}
rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index[x.index<=CUT]) for x in D.values()])):
 vals={a:F[a].get(dt,np.nan) for a in A}; good=np.array([v for v in vals.values() if np.isfinite(v)])
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A:
  if np.isfinite(vals[a]): sig.append((dt,a,vals[a]-med))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in D[a].index or not np.isfinite(vals[a]): continue
   i=D[a].index.get_loc(dt); j=i+h
   if j<len(D[a]) and D[a].index[j]<=CUT: f.append(vals[a]-med); y.append(D[a].close.iloc[j]/D[a].close.iloc[i]-1)
  if len(f)>=8:
   q=spearmanr(f,y).statistic
   if np.isfinite(q): rows.append((dt,h,q,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-25')]:
  z=x.set_index('date').loc[lo:hi].ic; print('REG',lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('coverage',round(len(sig)/(len(w)*15),4),'turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_volconfirm_reversal.csv',index=False)
