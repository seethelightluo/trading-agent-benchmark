import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
D={s:load(Path('../persistent/stock_data')/(s+'.csv')) for s in U}
px=pd.concat([D[s]['close'].rename(s) for s in U],axis=1).sort_index().loc[:'2027-03-24']
op=pd.concat([D[s]['open'].rename(s) for s in U],axis=1).reindex(px.index)
v=load(Path('../persistent/index_data/VIX.csv'))['close'].reindex(px.index).ffill()
shock=v.pct_change(5).clip(-.5,.5).fillna(0)
f=-(px/op-1)*(1+2*shock.values[:,None]); f=f.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h); s=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(s); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
f.rename_axis('date').to_csv('scripts/miner_1_20270325_vix_shock_reversal_signal.csv')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(U))
