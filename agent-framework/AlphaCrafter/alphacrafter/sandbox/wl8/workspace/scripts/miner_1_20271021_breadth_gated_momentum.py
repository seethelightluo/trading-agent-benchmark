import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-20')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change();
# Observable at t: breadth of prior completed 20-day asset returns; signal is lagged one day.
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1)
# gated trend: reward 10d momentum when breadth is broad, invert when breadth is weak
mom=px.pct_change(10).shift(1); gate=np.where(breadth.shift(1)>=0.60,1,np.where(breadth.shift(1)<=0.40,-1,0))
sig=mom.mul(gate,axis=0); fwd=px.shift(-1)/px-1
vals=[]; ns=[]; dates=[]
for i,d in enumerate(sig.index):
 g=pd.DataFrame({'s':sig.iloc[i],'f':fwd.iloc[i]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  vals.append(spearmanr(g.s,g.f).statistic); ns.append(len(g)); dates.append(d)
a=np.array(vals)
print('end',px.index.max().date(),'dates',len(a),'rows',sum(ns),'avg_names',round(np.mean(ns),2),'coverage',round(sig.notna().mean().mean(),4))
print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
y=pd.Series(dates).dt.year
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',pd.Series(dates)>=END-pd.Timedelta(days=180))]:
 z=a[m.values] if len(a) else np.array([]); print(q,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('gate counts',pd.Series(gate).value_counts().to_dict())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20271021_breadth_gated_momentum_signal.csv',index=False)
