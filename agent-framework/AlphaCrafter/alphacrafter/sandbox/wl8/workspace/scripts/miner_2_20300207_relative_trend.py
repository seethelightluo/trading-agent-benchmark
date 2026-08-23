import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2030-02-06')
frames={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); frames[s]=d
# Cross-asset residual trend: asset's lagged 20d return relative to same-day cross-sectional median,
# scaled by its lagged 20d volatility; all inputs lagged before forward return.
raw=[]
for s,d in frames.items():
 r=d.close.pct_change(); ret20=d.close.pct_change(20); vol20=r.rolling(20).std()
 raw.append(pd.DataFrame({'date':d.index,'symbol':s,'ret20':ret20,'vol20':vol20,
                          'fwd5':d.close.shift(-5)/d.close-1,'fwd10':d.close.shift(-10)/d.close-1,'fwd20':d.close.shift(-20)/d.close-1}))
x=pd.concat(raw).reset_index(drop=True)
x['med20']=x.groupby('date').ret20.transform('median')
x['signal']=((x.ret20-x.med20)/x.vol20).shift(1) # shift within concatenated symbols is unsafe; correct below
# recompute per symbol lag
x=x.sort_values(['symbol','date']); x['signal']=((x.ret20-x.med20)/x.vol20).groupby(x.symbol).shift(1); x=x.sort_values('date')
for h in [5,10,20]:
 out=[]
 for dt,g in x.groupby('date'):
  g=g[['signal',f'fwd{h}']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8: out.append((dt,spearmanr(g.signal,g[f'fwd{h}']).statistic,len(g)))
 ic=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); mean=ic.ic.mean(); sd=ic.ic.std(ddof=1)
 print('H',h,'dates',len(ic),'avg_n',ic.n.mean(),'coverage',x.signal.notna().mean(),'IC',mean,'ICIR',mean/sd,'hit',(ic.ic>0).mean())
 for a,b in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-02-06'),('2029-08-10','2030-02-06')]:
  q=ic.loc[a:b]
  if len(q): print(' regime',a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
 if h==10:
  x[['date','symbol','signal']].dropna().to_csv('scripts/miner_2_20300207_relative_trend_signal.csv',index=False)
# turnover
w=x.pivot(index='date',columns='symbol',values='signal'); rr=w.rank(axis=1,pct=True); print('turnover',((rr-rr.shift()).abs().mean(axis=1)).mean())
