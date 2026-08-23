import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=6000)
 except:pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=6000)
  except:d=None
 if d is not None:px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); res=r-r.mean(axis=1).values[:,None]; v=r.rolling(20,min_periods=10).std(); f=(-res.rolling(5,min_periods=5).sum()/v).shift(1)
print('universe',len(px),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 a=np.array(vals);print(f'{h}D dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={np.nanmean(a):.8f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252):.8f} hit={np.mean(a>0):.4f}')
rank=f.rank(axis=1,pct=True);print('coverage_all',f.notna().sum().sum()/(len(f)*15),'turnover',rank.diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20350215_volnorm_residual_reversal_signal.csv',index=False)
