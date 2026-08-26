import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=130: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); res=R.sub(R.mean(axis=1),axis=0)
a=res.rolling(5,min_periods=4).sum().shift(1); b=res.rolling(60,min_periods=45).sum().shift(1); v=res.rolling(30,min_periods=20).std().shift(1); f=((b/12-a)/(v*np.sqrt(5)+1e-8)).clip(-8,8); fw=P.shift(-10)/P-1
ics=[];ds=[];ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);ds.append(d);ns.append(len(z))
x=np.array(ics); di=pd.DatetimeIndex(ds); print('dates',len(x),'start',di[0].date(),'end',di[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(x.mean(),6),'ICIR_daily',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),6)); print('turnover',round(pd.DataFrame([f.loc[d].rank(pct=True) for d in ds]).diff().abs().mean().mean(),6)); pd.DataFrame([(d,s,float(f.loc[d,s])) for d in f.index for s in f.columns if pd.notna(f.loc[d,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_1_20340914_residual_accel_5d_signal.csv',index=False)
