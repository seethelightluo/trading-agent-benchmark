import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
    D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Drawdown-adjusted path efficiency: directional net return per absolute path,
# discounted by the worst peak-to-trough drawdown in the same 40-day window.
path=r.abs().rolling(40,min_periods=30).sum()
net=p.pct_change(40)
peak=p.rolling(40,min_periods=30).max()
dd=(p/peak-1).rolling(40,min_periods=30).min().abs()
f=(net/(path+1e-8)/(1+2*dd)).shift(1)
rows=[]
for i in range(len(p)-20):
    for h in [1,5,10,20]:
        z=pd.concat([f.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
        if len(z)>=8: rows.append((p.index[i],h,len(z),spearmanr(z.x,z.y).statistic))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('period',p.index.min().date(),p.index.max().date(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].set_index('date').ic
 print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for label,a,b in [('2020-25','2020-01-01','2025-12-31'),('2026+','2026-01-01','2031-04-03'),('2029+','2029-01-01','2031-04-03'),('2030+','2030-01-01','2031-04-03')]:
  q2=q.loc[a:b]
  if len(q2)>20: print(' ',label,'n',len(q2),'IC %.6f ICIR %.6f'%(q2.mean(),q2.mean()/q2.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
f.to_csv('scripts/miner_2_20310403_drawdown_path_efficiency_signal.csv')
