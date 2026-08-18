import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=2600).set_index('date')['close'].astype(float) for s in U}
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); look=40; h=10
rows=[]
for i in range(max(look,25),len(P)-h):
    vals={}; fw={}
    for s in U:
        vol=R[s].iloc[i-19:i+1].std()
        if np.isfinite(vol) and vol>0 and pd.notna(P[s].iloc[i-look]) and pd.notna(P[s].iloc[i+h]):
            vals[s]=(P[s].iloc[i]/P[s].iloc[i-look]-1)/vol
            fw[s]=P[s].iloc[i+h]/P[s].iloc[i]-1
    if len(vals)>=8:
        a=pd.Series(vals); b=pd.Series(fw).reindex(a.index)
        ic=a.corr(b)
        if np.isfinite(ic): rows.append((P.index[i],ic,len(vals)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avgN',z.n.mean(),'IC %.8f ICIR %.8f hit %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
print('coverage %.6f'%((z.n/len(U)).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2100')]:
 q=z.loc[a:b].ic
 print(a,b,'dates',len(q),'IC %.8f ICIR %.8f hit %.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
# causal signal artifact for audit
sig=[]
for d in z.index:
 i=P.index.get_loc(d); vv={}
 for s in U:
  vol=R[s].iloc[i-19:i+1].std()
  if vol>0 and pd.notna(P[s].iloc[i-look]): vv[s]=(P[s].iloc[i]/P[s].iloc[i-look]-1)/vol
 if len(vv)>=8:
  r=pd.Series(vv).rank(pct=True)
  for s,v in r.items(): sig.append((d,s,v))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20280121_risk_momentum40_signal.csv',index=False)
print('artifact rows',len(sig))
