import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>300: xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index(); r=p.pct_change()
# acceleration: medium trend minus short trend, contrarian to recent acceleration, volatility normalized
ret60=p/p.shift(60)-1; ret20=p/p.shift(20)-1
vol=r.rolling(60).std()*np.sqrt(60)
f=-(ret60-ret20)/(vol+1e-12)
# lag signal one day
f=f.shift(1)
rows=[]
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1
 ics=[]; n=[]; dates=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z)); dates.append(dt)
 q=pd.Series(ics).dropna(); print(h,'dates',len(q),'avg_n',round(np.mean(n),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
# signal turnover cross sectional rank changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),6),'turnover',round(turn,6),'rows',len(p),'instruments',len(xs),'end',p.index.max())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320805_acceleration_reversal_60d_signal.csv',index=False)
