import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
C=pd.DataFrame({s:D[s]['close'] for s in U}).sort_index(); cut=pd.Timestamp('2028-01-28')
C=C.loc[:cut]; lb,h=5,10
r=C.pct_change(lb); f=-(r.sub(r.median(axis=1),axis=0)); Y=C.shift(-h)/C-1
rows=[]
for d in f.index:
 g=pd.DataFrame({'f':f.loc[d],'y':Y.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
 if d>=pd.Timestamp('2020-01-01') and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  rows.append((d,spearmanr(g.f,g.y).statistic,len(g)))
z=np.array([x[1] for x in rows]); ds=pd.DatetimeIndex([x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('dates',len(z),'avgN',ns.mean(),'coverage',ns.mean()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-28')]:
 q=z[(ds>=pd.Timestamp(a))&(ds<=pd.Timestamp(b))]; print('regime',a,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
for h2 in [1,5,10,20]:
 yy=C.shift(-h2)/C-1; q=[]
 for d in f.index:
  g=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=np.array(q); print('decay',h2,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=[]
for d in f.index:
 g=pd.DataFrame({'signal':f.loc[d],'forward_return_10d':Y.loc[d]}).dropna()
 if len(g)>=8:
  for s,row in g.iterrows():out.append({'date':d,'symbol':s,**row.to_dict()})
pd.DataFrame(out).to_csv('scripts/miner_3_20280128_relative_reversal_5d10d_ic.csv',index=False)
