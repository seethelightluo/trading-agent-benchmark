import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=[]; vol=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150: d=get_index_daily_data(s,2600)
 if d is not None: px.append(d[['date','close']].assign(symbol=s)); vol.append(d[['date','volume']].assign(symbol=s))
w=pd.concat(px).pivot(index='date',columns='symbol',values='close').sort_index(); v=pd.concat(vol).pivot(index='date',columns='symbol',values='volume').reindex(w.index)
r=w.pct_change(); rv=r.rolling(20,min_periods=12).std(); vz=np.log1p(v).sub(np.log1p(v).rolling(40,min_periods=20).mean())/np.log1p(v).rolling(40,min_periods=20).std()
# liquidity-shock reversal: recent selloff is more likely to rebound when volume is unusually elevated, but scale by risk
f=(-r/(rv+1e-12))*(1+0.35*np.tanh(vz))
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1); f=f.where(f.ge(lo,axis=0),lo,axis=0).where(lambda x:x.le(hi,axis=0),hi,axis=0)
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns))
for h in [1,3,5,10]:
 ic=[];ns=[]; fr=w.shift(-h)/w-1
 for dt in w.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8: ic.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 q=pd.Series(ic).dropna();print('H',h,'n',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_3_20270325_liquidity_shock_reversal_signal.csv',index=False)
for a,b,c in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025+','2025-01-01','2099-12-31')]:
 z=[]
 for dt in f.loc[b:c].index:
  q=pd.concat([f.loc[dt],(w.shift(-1)/w-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('REG',a,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6))
