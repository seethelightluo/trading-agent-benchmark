import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 try:d=get_stock_daily_data(s,days=6000)
 except:F[s]=None;continue
 if d is not None and len(d):F[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close
F={k:v for k,v in F.items() if v is not None};p=pd.concat(F,axis=1).sort_index().ffill();r=p.pct_change(); ret=p.pct_change(60);vol=r.rolling(60).std()*np.sqrt(252);pos=r.gt(0).rolling(60).mean(); dd=(p/p.rolling(60).max()-1).rolling(60).min().abs(); f=(-(ret/(vol+.02))*(.5+pos)*(1-dd)).shift(1)
f.to_csv('scripts/miner_1_20340928_trend_persistence_contrarian_60d_signal.csv',index_label='date')
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1;z=[];n=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));n.append(len(a))
 z=pd.Series(z).dropna();print(f'H={h} dates={len(z)} avgN={np.mean(n):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
print('coverage',f.notna().mean(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'instruments',len(F))
