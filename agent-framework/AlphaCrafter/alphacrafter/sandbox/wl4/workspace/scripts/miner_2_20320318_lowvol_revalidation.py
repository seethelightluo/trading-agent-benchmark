import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv'); x.date=pd.to_datetime(x.date)
 D[s]=x.set_index('date').close.astype(float)
px=pd.concat(D,axis=1).sort_index().loc[:'2032-03-17']; r=np.log(px).diff()
v=r.rolling(40,min_periods=25).std(); vv=v.diff().rolling(20,min_periods=12).std()
f=(.7*(-v)+.3*(-vv)).shift(1)
for h in [5,10,20]:
  ics=[]; ns=[]; turnovers=[]; prev=None
  for dt in f.index:
    z=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
    if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    q=f.loc[dt].rank(pct=True)
    if prev is not None: turnovers.append((q-prev).abs().dropna().mean())
    prev=q
  a=np.asarray(ics); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.8f} ICIR={a.mean()/(a.std(ddof=1)/np.sqrt(len(a))):.5f} hit={np.mean(a>0):.5f} turnover={np.mean(turnovers):.5f}')
  if h==10:
   for n in [260,520,780]:
    q=a[-n:]; print(f'recent{n} IC={q.mean():.8f} ICIR={q.mean()/(q.std(ddof=1)/np.sqrt(len(q))):.5f} hit={np.mean(q>0):.5f}')
print('cutoff',px.index.max().date(),'assets',px.shape[1])
