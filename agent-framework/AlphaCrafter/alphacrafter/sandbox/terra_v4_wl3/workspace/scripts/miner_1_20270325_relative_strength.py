import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s): return pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
D={s:load(s) for s in U}
px=pd.concat([D[s]['close'].rename(s) for s in U],axis=1).sort_index().loc[:'2027-03-24']
# Relative-strength persistence: prior 20d return minus same-day cross-sectional median.
r=px.pct_change(20)
f=r.sub(r.median(axis=1),axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h); ic=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(ic,index=dates)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for label,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-03-24')]:
  q=s.loc[a:b]; print('REG',h,label,len(q),round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(U),'dates',len(px))
f.rename_axis('date').to_csv('scripts/miner_1_20270325_relative_strength_signal.csv')
