import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-11-17'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); bench=r.mean(axis=1)
# idiosyncratic trend: asset 20d return minus contemporaneous cross-asset benchmark trend, scaled by lagged residual volatility
res=r.sub(bench,axis=0); trend=res.rolling(20,min_periods=15).sum(); vol=res.rolling(20,min_periods=15).std(); sig=(trend/(vol+1e-12)).shift(1)
fwd=px.shift(-1)/px-1
A=[];N=[];D=[]
for d in sig.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  z=spearmanr(g.s,g.f).statistic
  if np.isfinite(z): A.append(z);N.append(len(g));D.append(d)
a=np.array(A); print('end',px.index.max().date(),'dates',len(a),'rows',sum(N),'avg_names',round(np.mean(N),2),'coverage',round(sig.notna().mean().mean(),4)); print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
y=pd.Series(D).dt.year
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',pd.Series(D)>=END-pd.Timedelta(days=180))]:
 z=a[m.values]; print(q,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20271118_residual_trend_strength_signal.csv',index=False)
