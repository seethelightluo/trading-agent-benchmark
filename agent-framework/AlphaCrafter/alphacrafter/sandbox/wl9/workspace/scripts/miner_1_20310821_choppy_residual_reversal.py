import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4000) for s in U}
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index()
r=np.log(px).diff(); dates=px.index
rr=r.rolling(60).sum(); res=rr.sub(rr.mean(axis=1),axis=0)
eff=r.abs().rolling(60).sum().div(rr.abs().replace(0,np.nan))
chop=(1-eff).clip(0,1); vol=r.rolling(40).std().replace(0,np.nan)
f=(-res/vol*(0.75+1.25*chop)).shift(1)
f.stack().rename('signal').to_csv('scripts/miner_1_20310821_choppy_residual_reversal_signal.csv')
rows=[]
for i in range(61,len(dates)-20):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+20]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dates[i],len(z),z.x.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(252),'hit',(a.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10,20,40]:
 rr2=[]
 for i in range(61,len(dates)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rr2.append(z.x.corr(z.y))
 print('decay',h,np.nanmean(rr2),len(rr2))
for y in [2027,2028,2029,2030,2031]:
 q=a[a.index.year==y].ic
 print('regime',y,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
