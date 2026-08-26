import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in u:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None:P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
pd_=pd.DataFrame(P).sort_index().ffill(); r=pd_.pct_change()
# recovery efficiency: medium return per unit downside deviation, lagged one day
mean=r.rolling(30,min_periods=25).mean(); down=r.where(r<0,0).rolling(30,min_periods=25).std()
f=(pd_/pd_.shift(10)-1)/(down*np.sqrt(30)+1e-8)
for h in [1,5,10,20]:
 fr=pd_.shift(-h)/pd_-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);a=a[np.isfinite(a)]; print('H%d IC %.6f ICIR %.6f hit %.4f dates %d thirds %s'%(h,a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),len(a),[round(x,6) for x in map(np.mean,np.array_split(a,3))]))
print('assets',len(u),'dates',len(pd_),'coverage',round(f.notna().sum(axis=1).mean()/len(u),4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20330516_recovery_efficiency_signal.csv',index=False)
