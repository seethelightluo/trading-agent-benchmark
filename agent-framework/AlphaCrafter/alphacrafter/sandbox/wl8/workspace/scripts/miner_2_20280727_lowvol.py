import pandas as pd,numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): P[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); sig=-r.rolling(20,min_periods=15).std().shift(1)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']); q=z.ic
 print('h',h,'dates',len(q),'avgN',z.n.mean(),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),sig.notna().sum().sum()/sig.size))
 for y in [2020,2021,2022,2023,2024,2025,2026,2027,2028]:
  a=z[z.date.dt.year==y].ic
  if len(a): print(y,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),4))
rank=sig.rank(axis=1,pct=True); print('turn',rank.diff().abs().mean(axis=1).mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280727_lowvol_signal.csv',index=False)
