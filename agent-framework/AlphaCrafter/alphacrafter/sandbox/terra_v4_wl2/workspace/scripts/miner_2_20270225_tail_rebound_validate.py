import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
f={}
for a in A:
 dd=p[a]/p[a].rolling(20,min_periods=20).max()-1
 f[a]=(-p[a].pct_change(3)).where(dd < -0.08)
out=pd.concat(f,axis=1).rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal')
out.to_csv('../persistent/factor_signals_miner_2_20270225_tail_rebound.csv',index=False)
# Validate next-day paper IC, with >=8 valid names.
from scipy.stats import spearmanr
rows=[]
for d in out.date.sort_values().unique():
 vals=[]
 for a in A:
  if d in p[a].index and pd.notna(f[a].get(d,np.nan)):
   y=p[a].pct_change().shift(-1).get(d,np.nan)
   if pd.notna(y): vals.append((f[a].loc[d],y))
 if len(vals)>=8:
  z=pd.DataFrame(vals,columns=['x','y'])
  if z.x.nunique()>1 and z.y.nunique()>1: rows.append((d,spearmanr(z.x,z.y).statistic,len(z)))
q=np.array([x[1] for x in rows]); print('dates',len(q),'avg_n',np.mean([x[2] for x in rows]),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',out.signal.notna().mean(),'date_coverage',len(q)/len(p[A[0]]))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 v=np.array([x[1] for x in rows if lo<=str(x[0])[:4]<=hi]); print(lo,len(v),v.mean() if len(v) else np.nan,v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
r=out.pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1); print('turnover',r.mean())
print('artifact','../persistent/factor_signals_miner_2_20270225_tail_rebound.csv')
