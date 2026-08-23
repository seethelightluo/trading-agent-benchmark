import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2030-04-17'); a=[]
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); r=d.close.pct_change(); dn=r.clip(upper=0).rolling(30).std()
 a.append(pd.DataFrame({'date':d.index,'symbol':s,'sig':(d.close.pct_change(60)/dn).shift(1),'fwd5':d.close.shift(-5)/d.close-1,'fwd10':d.close.shift(-10)/d.close-1,'fwd20':d.close.shift(-20)/d.close-1}))
x=pd.concat(a,ignore_index=True).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20]:
 z=[]
 for dt,g in x.groupby('date'):
  q=g[['sig',f'fwd{h}']].dropna()
  if len(q)>=8:
   v=spearmanr(q.sig,q[f'fwd{h}']).statistic
   if np.isfinite(v): z.append((dt,v,len(q)))
 ic=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); ir=m/ic.ic.std(ddof=1)
 print(f'H{h} dates={len(ic)} avg_n={ic.n.mean():.2f} coverage={x.sig.notna().mean():.4f} IC={m:.6f} ICIR={ir:.6f} hit={(ic.ic>0).mean():.4f}')
 for p,q in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029','2030-04-17'),('2029-10-01','2030-04-17')]:
  t=ic.loc[p:q]
  if len(t)>2: print('REG',p,q,len(t),f'{t.ic.mean():.6f}',f'{t.ic.mean()/t.ic.std(ddof=1):.6f}')
 if h==10:x[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20300418_downside_momentum_signal.csv',index=False)
print('assets',x.symbol.nunique(),'dates',x.date.nunique())
