import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-31')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change(); common=r.mean(axis=1)
# Residual momentum: trailing 20-session asset return after removing rolling beta exposure to equal-weight common return.
w=60; mom= p.pct_change(20); f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(w,len(p)):
    x=r.iloc[i-w:i]; z=common.iloc[i-w:i]; vz=z.var()
    if pd.notna(vz) and vz>1e-12:
        beta=x.apply(lambda q:q.cov(z)/vz)
        f.iloc[i]=mom.iloc[i]-beta*common.iloc[i-20:i+1].sum()
rows=[]
for i in range(len(p)-1):
    q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
    if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: rows.append((p.index[i],spearmanr(q.f,q.y).statistic,len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']); z=a.ic.values
print('dates',len(z),'avg_names',a.n.mean(),'coverage',len(z)/(len(p)-1),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31')]:
 v=a[(a.date>=lo)&(a.date<=hi)].ic; print(label,len(v),v.mean(),v.mean()/v.std(ddof=1))
for h in [3,5,10]:
 vals=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic)
 v=np.array(vals); print('horizon',h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
# signal artifact for audit
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20261231_resid_momentum.csv',index=False)
