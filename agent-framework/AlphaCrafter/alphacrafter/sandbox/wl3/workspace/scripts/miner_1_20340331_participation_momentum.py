import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 return d[['date','close','volume']].copy() if d is not None else None
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
px=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
vol=pd.concat({s:d.set_index('date').volume for s,d in D.items()},axis=1).sort_index().reindex(px.index).ffill()
r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
# medium-term residual momentum confirmed by unusually strong participation; all inputs lagged
mom=res.rolling(15).sum(); vs=(vol/(vol.rolling(60).median()+1e-12)).clip(0.5,3).apply(np.log)
confirm=(1+0.35*vs.rolling(10).mean()).clip(0.6,1.4)
f=mom.mul(confirm,axis=0).shift(1)
fr=px.shift(-10)/px-1; rows=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic):rows.append((dt,ic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mean=q.ic.mean();sd=q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',len(q)/len(px),'IC10',mean,'ICIR10',mean/sd*np.sqrt(252/10),'hit',(q.ic>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 z=q.tail(n);print('recent',n,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252/10))
for h in [5,20]:
 yy=px.shift(-h)/px-1;a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/h))
f.to_csv('scripts/miner_1_20340331_participation_momentum_signal.csv',index_label='date');q.to_csv('scripts/miner_1_20340331_participation_momentum_ic.csv')
