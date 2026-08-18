import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 d=pd.read_csv(os.path.join(base,a+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d
p=pd.DataFrame(px).sort_index()
# one candidate: breadth-conditioned 20d trend, causal market breadth is median lagged 20d returns
r20=p.pct_change(20)
breadth=r20.median(axis=1)
f=r20.mul(np.where(breadth>=0,1.0,-1.0),axis=0)
# evaluate 10d forward, only dates with >=8 names
fr=p.shift(-10)/p-1
rows=[]
for dt in p.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  rows.append((dt,ic,ok.mean(),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','coverage','n']).set_index('date')
z=z.replace([np.inf,-np.inf],np.nan).dropna()
mean=z.ic.mean(); sd=z.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
# signal turnover as average rank changes / cross-section; daily rank turnover
ranks=f.rank(axis=1,pct=True)
to=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean()
print('dates',len(z),'avgN',z.n.mean(),'IC',mean,'ICIR',icir,'hit', (z.ic>0).mean(),'coverage',z.coverage.mean(),'turnover',to)
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-05-08')]:
 q=z.loc[lo:hi].ic
 print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan,'hit',(q>0).mean())
# save complete signal artifact for provenance
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20280509_breadth_conditioned_trend20_signal.csv')
