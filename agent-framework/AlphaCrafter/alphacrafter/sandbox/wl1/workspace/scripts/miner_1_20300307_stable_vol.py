import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof=pd.Timestamp('2030-03-07'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for s in syms:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):p[s]=d[d.date<=asof].set_index('date').close.astype(float)
c=pd.DataFrame(p).reindex(columns=syms); r=c.pct_change()
# Stable-volatility carry: inverse medium realized volatility, with a mild downside-volatility penalty.
vol=r.rolling(40).std()*np.sqrt(252); down=r.where(r<0,0).rolling(40).std()*np.sqrt(252)
f=1/(vol+0.05)-0.35*down
rows=[]
for dt in f.index:
 for h in [1,5,10,20]:
  z=pd.concat([f.loc[dt],(c.shift(-h)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 x=q[q.h==h].set_index('date'); print('H',h,'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for a,b in [('2020','2025-12-31'),('2026','2028-12-31'),('2029','2029-12-31'),('2030','2030-03-07')]:
  y=x[(x.index>=a)&(x.index<=b)]
  if len(y)>2:print(' ',a,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300307_stable_vol_signal.csv',index=False)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'coverage',round(out.symbol.nunique()/15,4),'rows',len(out))
