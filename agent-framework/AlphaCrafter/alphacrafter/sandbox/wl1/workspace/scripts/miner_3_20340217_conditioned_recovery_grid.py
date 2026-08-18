import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change(); down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(40);vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
rec=-(np.log(px/px.shift(60))-.70*np.log(px/px.shift(10)))/(down+.5*vol+1e-6)
eff=(px.pct_change(20)/(r.abs().rolling(20).sum()+1e-12))/(r.rolling(20).std()+1e-12);er=eff.rank(axis=1,pct=True)
for alpha in [-1,-.5,.5,1,2,3]:
 f=(rec*(1+alpha*(.5-er))).shift(1);fr=px.pct_change(10).shift(-10);z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=np.array(z);print('alpha',alpha,'dates',len(z),'IC %.8f ICIR %.8f hit %.4f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)))
 if alpha==2:
  out=f.copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_3_20340217_conditioned_recovery_signal.csv',index=False)
