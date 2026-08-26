import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
# Relative shock reversal: asset's 5d return relative to contemporaneous
# cross-sectional median, scaled by idiosyncratic 30d volatility; fade only
# when cross-sectional dispersion is elevated. Signal is lagged one session.
csmed=r.median(axis=1); disp=r.sub(csmed,axis=0).abs().median(axis=1)
threshold=disp.rolling(80,min_periods=40).median()
rv=r.rolling(30,min_periods=20).std().replace(0,np.nan)
raw=-(np.log(p/p.shift(5)).sub(np.log(p/p.shift(5)).median(axis=1),axis=0)).div(rv)
gate=(disp>threshold).astype(float)
f=raw.mul(gate,axis=0).shift(1).clip(-8,8)
for h in [1,5,10,20]:
 qs=[]; ns=[]
 fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(qs).dropna(); ic=q.mean(); ir=ic/q.std(ddof=1)*np.sqrt(252)
 print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(q>0).mean():.4f}')
 if h==1: q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20301202_relative_shock_reversal_ic.csv',index=False)
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F),'first',p.index.min().date(),'last',p.index.max().date(),'activation',gate.mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20301202_relative_shock_reversal_signal.csv',index=False)
