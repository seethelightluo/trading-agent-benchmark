import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
P={}
for a in A:
 p=f'{B}/{a}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index(); R=P.pct_change()
# Trend quality: signed 20-session return divided by path length; lagged one day.
# A cross-sectional breadth gate boosts quality trends when market breadth is constructive.
path=R.abs().rolling(20,min_periods=15).sum(); raw=P.pct_change(20)/path
breadth=(R>0).sum(axis=1)/R.notna().sum(axis=1)
gate=(0.75+0.5*breadth.rolling(15,min_periods=10).mean()).clip(.75,1.25)
F=raw.mul(gate,axis=0).shift(1)
rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('candidate efficiency_trend_quality_20d'); print('dates',len(r),'avgN',round(r.n.mean(),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),6))
for n in [260,520,780]:
 q=s.tail(min(n,len(s))); print('recent',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20330721_efficiency_trend_quality_20d_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20330721_efficiency_trend_quality_20d_signal.csv')
