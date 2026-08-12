import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change()
for name,mode in [('tail',0),('volshock',1),('breadth',2)]:
 rows=[]
 for t in range(65,len(P)-11):
  r3=R.iloc[t-2:t+1].sum();v=R.iloc[t-19:t+1].std(ddof=1); med=float(r3.median()); mean=float(r3.mean())
  if mode==0: gate=med < r3.quantile(.35)
  elif mode==1: gate=(r3.abs()/v).median()>1.0
  else: gate=mean < -.01
  f=-(r3/v.replace(0,np.nan)); f=f*(1.3 if gate else .7);f=f.replace([np.inf,-np.inf],np.nan).dropna()
  fw=R.iloc[t+1:t+2].sum().reindex(f.index);q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8:rows.append(q.iloc[:,0].corr(q.iloc[:,1]))
 a=pd.Series(rows).dropna();print(name,'dates',len(a),'N',len(U),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
