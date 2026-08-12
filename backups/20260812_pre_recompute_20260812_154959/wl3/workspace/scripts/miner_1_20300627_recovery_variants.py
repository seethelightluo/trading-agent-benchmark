import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 D[s]=d.set_index("date").close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff(); res=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0); vol=r.rolling(20).std()*np.sqrt(20); dd=px/px.rolling(60).max()-1
variants={"plain":-res/(vol+1e-12),"ddboost":-res/(vol+1e-12)*(1+(-dd).clip(0,.35)/.35),"ddclip":-res/(vol+1e-12)*(1+(-dd).clip(0,.20)/.20),"downrisk":-res/(r.where(r<0,0).rolling(20).std()*np.sqrt(20)+1e-12)}
for name,f in variants.items():
 f=f.shift(1); q=[]
 for i,dt in enumerate(px.index[:-5]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+5]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print(name,len(q),"IC %.6f ICIR %.6f hit %.4f"%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()))
 if name=="plain": f.stack().rename("signal").reset_index().rename(columns={"level_0":"date","level_1":"symbol"}).to_csv("scripts/miner_1_20300627_recovery_plain_signal.csv",index=False)
