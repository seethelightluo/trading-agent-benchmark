import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2030-12-26')
P={}; V={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); P[s]=d.close.astype(float); V[s]=d.volume.astype(float).replace(0,np.nan)
P=pd.DataFrame(P); V=pd.DataFrame(V)
r=P.pct_change(); m=r.mean(axis=1); beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=(r.rolling(20,min_periods=15).sum()-beta.rolling(20,min_periods=15).mean().mul(m.rolling(20,min_periods=15).sum(),axis=0))
vol=np.log(V.rolling(10,min_periods=5).mean()/V.rolling(60,min_periods=20).mean())
fac=(res*(1+0.30*np.tanh(vol))).shift(1); rows=[]
for h in [1,5,10,20]:
 fwd=P.shift(-h)/P-1
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n']); print('panel_end',P.index.max().date(),'dates',len(P.index),'assets',len(U))
for h in [1,5,10,20]:
 q=r[r.h==h]; mu=q.ic.mean(); sd=q.ic.std(ddof=1); ir=mu/sd*np.sqrt(252); recent=q.tail(260); rm=recent.ic.mean(); ri=rm/recent.ic.std(ddof=1)*np.sqrt(252)
 print(f'h{h}: IC={mu:.6f} ICIR={ir:.6f} hit={(q.ic>0).mean():.4f} dates={len(q)} avgN={q.n.mean():.2f} recent260={rm:.6f}/{ri:.6f}')
print('coverage',fac.notna().sum(axis=1).mean()/len(U),'rank_turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
