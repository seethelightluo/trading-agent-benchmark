import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,2600)
    except Exception: pass
    if d is None:
        try: d=get_stock_daily_data(s,2600)
        except Exception: pass
    if d is not None: fs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(fs).sort_index().ffill(); r=p.pct_change(); mkt=r.mean(axis=1)
# residualized 20-day trend: asset return minus rolling 60-day beta times common-market return
cov=r.rolling(60,min_periods=30).cov(mkt); var=mkt.rolling(60,min_periods=30).var()
beta=cov.div(var.replace(0,np.nan),axis=0); resid20=(r.rolling(20,min_periods=15).sum()-beta.rolling(20,min_periods=15).mean().mul(mkt.rolling(20,min_periods=15).sum(),axis=0))
vol=r.rolling(20,min_periods=12).std()*np.sqrt(10)
f=(resid20/vol.replace(0,np.nan)).shift(1)
def calc(h):
 rows=[]
 for i in range(80,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append([p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')])
 return rows
for h in [5,10,20]:
 rows=calc(h); a=np.array([x[2] for x in rows]); print('horizon',h,'dates',len(a),'avg_n',np.mean([x[1] for x in rows]),'coverage',np.mean([x[1] for x in rows])/len(fs),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 if h==10: pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_1_20290618_residual_momentum_signal.csv',index=False)
print('assets',len(fs),'period',p.index.min(),p.index.max())
