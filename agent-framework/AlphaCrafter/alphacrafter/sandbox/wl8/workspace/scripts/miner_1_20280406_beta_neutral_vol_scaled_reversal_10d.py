import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-04-05')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=r.rolling(10,min_periods=10).sum()-beta.mul(m.rolling(10,min_periods=10).sum(),axis=0)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(-res/vol).shift(1); f.to_csv('scripts/miner_1_20280406_beta_neutral_vol_scaled_reversal_10d_signal.csv',index_label='date')
fwd={h:px.shift(-h)/px-1 for h in [1,3,5,10]}
for h,y in fwd.items():
  ics=[]; dates=[]; ns=[]
  for d in px.index:
    g=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
    if len(g)>=8:
      q=spearmanr(g.f,g.y).statistic
      if np.isfinite(q): ics.append(q); dates.append(d); ns.append(len(g))
  a=np.asarray(ics); print('h',h,'dates',len(a),'rows',sum(ns),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
  if h==1:
   for lab,fn in [('2026',lambda d:d.year==2026),('2027',lambda d:d.year==2027),('2028',lambda d:d.year>=2028),('recent180',lambda d:d>=END-pd.Timedelta(days=180))]:
    z=a[[i for i,d in enumerate(dates) if fn(d)]]
    print(lab,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
