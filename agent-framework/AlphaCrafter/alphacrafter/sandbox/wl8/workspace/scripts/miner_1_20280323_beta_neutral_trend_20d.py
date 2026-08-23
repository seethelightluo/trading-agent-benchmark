import numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-22')
P={}
for s in ASSETS:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
r20=px.pct_change(20); mr20=m.rolling(20,min_periods=15).sum(); f=(r20-beta.mul(mr20,axis=0)).shift(1)
fwd=px.shift(-1)/px-1; ics=[]; dates=[]; ns=[]
for d in px.index:
 g=pd.DataFrame({'f':f.loc[d],'y':fwd.loc[d]}).dropna()
 if len(g)>=8:
  q=g.f.corr(g.y,method='spearman')
  if np.isfinite(q):ics.append(q);dates.append(d);ns.append(len(g))
a=np.array(ics); print('factor beta_neutral_20d_trend'); print('dates',len(a),'rows',sum(ns),'avgN',round(np.mean(ns),2),'coverage',round(f.notna().sum().sum()/f.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lab,fn in [('2020-22',lambda d:d.year<=2022),('2023-25',lambda d:2023<=d.year<=2025),('2026',lambda d:d.year==2026),('2027+',lambda d:d.year>=2027),('recent180',lambda d:d>=END-pd.Timedelta(days=180))]:
 z=a[[i for i,d in enumerate(dates) if fn(d)]]; print(lab,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
for h in [1,3,5]:
 y=px.shift(-h)/px-1; z=[]
 for d in px.index:
  g=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(g)>=8:z.append(g.f.corr(g.y,method='spearman'))
 z=np.array(z); print('h',h,'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6),'dates',len(z))
f.to_csv('scripts/miner_1_20280323_beta_neutral_trend_20d_signal.csv',index_label='date')
