import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2030-04-17'); rows=[]
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date')
 r=d.close.pct_change(); r20=d.close.pct_change(20); r60=d.close.pct_change(60)
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'r20':r20,'r60':r60,'fwd5':d.close.shift(-5)/d.close-1,'fwd10':d.close.shift(-10)/d.close-1,'fwd20':d.close.shift(-20)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
# Market breadth is formed from same-day completed returns, then signal is lagged one day.
b=x.groupby('date').r20.median().rename('breadth')
x=x.join(b,on='date'); x['raw']=(x.r20+x.r60)/2 * np.sign(x.breadth)
x['med']=x.groupby('date').raw.transform('median'); x['signal']=(x.raw-x.med).groupby(x.symbol).shift(1)
for h in [5,10,20]:
 out=[]
 for dt,g in x.groupby('date'):
  z=g[['signal',f'fwd{h}']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.signal,z[f'fwd{h}']).statistic,len(z)))
 ic=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); ir=m/ic.ic.std(ddof=1)
 print(f'H{h} dates={len(ic)} avg_n={ic.n.mean():.2f} coverage={x.signal.notna().mean():.4f} IC={m:.6f} ICIR={ir:.6f} hit={(ic.ic>0).mean():.4f}')
 for a,bnd in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-04-17'),('2029-10-01','2030-04-17')]:
  q=ic.loc[a:bnd]
  if len(q)>2: print('REG',a,bnd,len(q),f'{q.ic.mean():.6f}',f'{q.ic.mean()/q.ic.std(ddof=1):.6f}')
 if h==10: x[['date','symbol','signal']].dropna().to_csv('scripts/miner_2_20300418_breadth_conditioned_trend_signal.csv',index=False)
w=x.pivot(index='date',columns='symbol',values='signal'); print('assets',x.symbol.nunique(),'dates',x.date.nunique(),'turnover_proxy',((w.rank(axis=1,pct=True)-w.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
