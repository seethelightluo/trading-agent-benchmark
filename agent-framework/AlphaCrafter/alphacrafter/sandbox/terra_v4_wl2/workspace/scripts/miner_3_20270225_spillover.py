import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:np.nan for a in A}
 for a in A:
  others=r.loc[dt].drop(labels=a,errors='ignore') if dt in r.index else pd.Series(dtype=float)
  vals[a]=others.median() if others.notna().sum()>=7 else np.nan
  sig.append((dt,a,vals[a]))
 f=[];y=[]
 for a in A:
  if dt not in p[a].index or not np.isfinite(vals[a]):continue
  i=p[a].index.get_loc(dt)
  if i+1<len(p[a]):f.append(vals[a]);y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
 if len(f)>=8:rows.append((dt,spearmanr(f,y).statistic,len(f)))
x=pd.DataFrame(rows,columns=['date','ic','n']);print('dates',len(x),'avg_n',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=x.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std())
o=pd.DataFrame(sig,columns=['date','asset','signal']);o.to_csv('../persistent/factor_signals_miner_3_20270225_spillover.csv',index=False);print('turn',o.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
