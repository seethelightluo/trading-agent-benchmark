import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv("../persistent/stock_data/"+s+".csv")
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date); P[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# interpretable candidate: medium momentum scaled by recent downside risk; PIT lag via factor at t, forward from t+1
f=px.pct_change(20)/(r.where(r<0).rolling(20).std().replace(0,np.nan))
# normalize cross section, evaluate forward horizons
print('dates',px.index.min(),px.index.max(),'assets',len(px.columns))
for h in [1,5,10]:
 vals=[]
 for i in range(len(px)-h):
  a=f.iloc[i]; y=px.pct_change(h).shift(-h).iloc[i] # wrong includes t+h/t? shift gives return at i+h, use px[i+h]/px[i]-1
  y=px.iloc[i+h]/px.iloc[i]-1
  z=pd.concat([a,y],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=pd.Series(vals).dropna(); print('h',h,'n',len(x),'avgN',len(U),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit', (x>0).mean())
# turnover cross-sectional rank changes
rr=f.rank(axis=1,pct=True); turn=(rr.diff().abs().mean(axis=1)).mean(); print('turn',turn,'coverage',f.notna().sum(axis=1).mean()/len(U))
# save signal artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','asset','signal']; out.to_csv('../persistent/factor_signals_miner_2_20270225_downside_scaled_mom20.csv',index=False)
