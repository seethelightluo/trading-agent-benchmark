import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); D[s]=d[d.index<=end]
def calc(s):
 d=D[s]; return -(d.open/d.close.shift(1)-1) # gap fade, prior completed day
F={s:calc(s) for s in U}
for h in [1,5,10]:
 out=[]
 for dt in D[U[0]].index:
  a=[]
  for s in U:
   d=D[s]; f=F[s]
   if dt in d.index:
    j=d.index.get_loc(dt)
    if j+h<len(d) and pd.notna(f.loc[dt]): a.append((f.loc[dt],d.close.iloc[j+h]/d.close.iloc[j]-1))
  if len(a)>=8: out.append((dt,spearmanr(*np.array(a).T).statistic,len(a)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean(),q.n.mean()/15)
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  x=q.loc[a:b].ic; print(a,b,round(x.mean(),4),round(x.mean()/x.std(),3))
print('turnover',pd.DataFrame(F).rank(pct=True).diff().abs().mean(axis=1).mean())
