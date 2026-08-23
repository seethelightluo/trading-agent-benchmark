import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2030-03-06'); rows=[]
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); r=d.close.pct_change(); ret=d.close.pct_change(20); vol=r.rolling(20).std()
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'ret':ret,'vol':vol,'fwd10':d.close.shift(-10)/d.close-1,'fwd5':d.close.shift(-5)/d.close-1,'fwd20':d.close.shift(-20)/d.close-1}))
x=pd.concat(rows,ignore_index=True).sort_values(['symbol','date']); x['med']=x.groupby('date').ret.transform('median'); x['signal']=((x.ret-x.med)/x.vol).groupby(x.symbol).shift(1); x=x.sort_values('date')
for h in [5,10,20]:
 out=[]
 for dt,g in x.groupby('date'):
  z=g[['signal',f'fwd{h}']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.signal,z[f'fwd{h}']).statistic,len(z)))
 ic=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); ir=m/ic.ic.std(ddof=1)
 print(f'H{h} dates={len(ic)} avg_n={ic.n.mean():.2f} coverage={x.signal.notna().mean():.4f} IC={m:.6f} ICIR={ir:.6f} hit={(ic.ic>0).mean():.4f}')
 for a,b in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-03-06')]:
  q=ic.loc[a:b]
  if len(q)>2: print('REG',a,b,len(q),f'{q.ic.mean():.6f}',f'{q.ic.mean()/q.ic.std(ddof=1):.6f}')
 if h==10: x[['date','symbol','signal']].dropna().to_csv('scripts/miner_2_20300307_trend_persistence_signal.csv',index=False)
print('assets',x.symbol.nunique(),'dates',x.date.nunique())
