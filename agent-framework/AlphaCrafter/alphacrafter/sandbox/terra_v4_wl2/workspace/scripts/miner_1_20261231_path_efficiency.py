import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-30')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
R=pd.DataFrame({s:p.pct_change() for s,p in P.items()})
y=pd.DataFrame({s:p.shift(-1)/p-1 for s,p in P.items()})
for w in [10,20,30,60]:
 # directional efficiency: signed net move divided by total path length
 f=R.rolling(w,min_periods=w).sum()/R.abs().rolling(w,min_periods=w).sum()
 vals=[]; dates=[]; ns=[]; ranks=[]
 for dt in f.index:
  g=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   vals.append(spearmanr(g.f,g.y).statistic); dates.append(dt); ns.append(len(g)); ranks.append(g.f.rank(pct=True))
 z=np.array(vals); rr=pd.DataFrame(ranks,index=dates)
 print(f'w={w} dates={len(z)} avg_names={np.mean(ns):.2f} coverage={f.notna().sum(axis=1).mean()/15:.2%} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={np.mean(z>0):.4f} turnover={rr.diff().abs().mean().mean():.4f}')
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=z[[lo<=d.year<=hi for d in dates]]
  print(' regime',lo,hi,'dates',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
 for h in [5,10]:
  yy=pd.DataFrame({s:p.shift(-h)/p-1 for s,p in P.items()}); zz=[]
  for dt in f.index:
   g=pd.DataFrame({'f':f.loc[dt],'y':yy.loc[dt]}).dropna()
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: zz.append(spearmanr(g.f,g.y).statistic)
  zz=np.array(zz); print(' horizon',h,'dates',len(zz),'IC',zz.mean(),'ICIR',zz.mean()/zz.std(ddof=1))
# artifact for best candidate (20d)
f=R.rolling(20,min_periods=20).sum()/R.abs().rolling(20,min_periods=20).sum()
f.to_csv('scripts/miner_1_20261231_path_efficiency_signal.csv',index_label='date')
