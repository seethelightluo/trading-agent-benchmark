import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:get_stock_daily_data(a,days=2400).set_index('date').close.astype(float) for a in A}
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# Relative volatility: reward lower realized risk while controlling for each day's cross-sectional risk regime.
vol=r.rolling(20,min_periods=15).std(); med=vol.median(axis=1); f=-(vol.div(med,axis=0)-1.0)
y=sum(r.shift(-k) for k in range(1,6))
rows=[]; turnovers=[]; prev=None
for i in range(len(p)-5):
 q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((p.index[i],q.f.corr(q.y),len(q)))
 z=f.iloc[i].rank(pct=True)
 if prev is not None: turnovers.append(np.nanmean(abs(z-prev)))
 prev=z
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=z.ic
print('assets',len(A),'dates',len(p),'valid_dates',len(z),'avg_names',z.n.mean(),'coverage',z.n.mean()/15)
print('daily IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turnover',np.nanmean(turnovers))
for w in [60,120,252]:
 q=x.tail(w);print('recent',w,q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0))
for yr,g in z.groupby(z.index.year):
 q=g.ic;print('year',yr,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10]:
 yy=sum(r.shift(-k) for k in range(1,h+1)); rr=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],yy.iloc[i]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rr.append(q.iloc[:,0].corr(q.iloc[:,1]))
 rr=np.array(rr);print('horizon',h,'dates',len(rr),'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1))
print('signal_corr_raw_vol',pd.concat([f.stack(),(-vol.div(med,axis=0)).stack()],axis=1).dropna().corr().iloc[0,1])
