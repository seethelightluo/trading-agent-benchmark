import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A})
# Downside-risk-adjusted medium-term momentum; lagged 20d return divided by 20d downside deviation.
down=r.where(r<0,0).rolling(20,min_periods=10).std()
ret=pd.DataFrame({a:p[a].pct_change(20) for a in A})
F=ret/(down+1e-8)
rows=[];sig=[]
for d in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:F.at[d,a] if d in F.index else np.nan for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A:sig.append((d,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in p[a].index or not np.isfinite(vals[a]):continue
   i=p[a].index.get_loc(d)
   if i+h<len(p[a]):f.append(vals[a]-med);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
x=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=x[x.h==h];print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=q.set_index('date').loc[lo:hi].ic;print(lo,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
o=pd.DataFrame(sig,columns=['date','asset','signal']);o.to_csv('../persistent/factor_signals_miner_1_20270225_downside_mom.csv',index=False)
print('turnover',o.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
