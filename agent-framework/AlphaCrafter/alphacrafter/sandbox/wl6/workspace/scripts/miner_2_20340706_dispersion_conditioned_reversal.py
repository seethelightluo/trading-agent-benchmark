import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 D[s]=x[['date','close']].drop_duplicates('date').set_index('date').sort_index()
p=pd.DataFrame({s:D[s].close.astype(float) for s in D}).sort_index().loc[:'2034-07-05']
r=p.pct_change(); r10=p.pct_change(10); vol=r.rolling(20).std()*np.sqrt(20)
res=r10.sub(r10.median(axis=1),axis=0)
disp=r10.std(axis=1).rolling(60).rank(pct=True)
active=(disp>0.50)
f=(-res/(vol+1e-12)).where(active).shift(1)
f.to_csv('scripts/miner_2_20340706_dispersion_conditioned_reversal_signal.csv',index_label='date')
for H in [5,10,20,40]:
 fr=p.pct_change(H).shift(-H); vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  if not bool(active.get(dt,False)): continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  q=f.loc[dt].dropna().rank(pct=True)
  if prev is not None:
   c=q.index.intersection(prev.index)
   if len(c)>=8: turns.append(np.mean(abs(q[c]-prev[c])))
  if len(q):prev=q
 a=np.array(vals); sd=np.nanstd(a,ddof=1)
 print(f'H={H} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={np.nanmean(a):.9f} ICIR={np.nanmean(a)/sd*np.sqrt(len(a)):.9f} hit={np.mean(a>0):.4f} turnover={np.mean(turns):.5f}')
fr=p.pct_change(10).shift(-10)
for label,lo,hi in [('2020-24',2020,2025),('2025-29',2025,2030),('2030-32',2030,2033),('2033-34',2033,2035)]:
 a=[]
 for dt in f.index:
  if not(lo<=dt.year<hi) or not bool(active.get(dt,False)): continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(f'REGIME {label} dates={len(a)} IC={np.mean(a) if a else np.nan:.6f}')
