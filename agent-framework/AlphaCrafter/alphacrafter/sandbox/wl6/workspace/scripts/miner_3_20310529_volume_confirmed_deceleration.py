import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}; volume={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d.assign(date=pd.to_datetime(d.date)).set_index('date')
  close[s]=x.close; volume[s]=x.volume.replace(0,np.nan)
P=pd.DataFrame(close).sort_index(); V=pd.DataFrame(volume).reindex(P.index).sort_index()
r=P.pct_change(); rv=r.rolling(40,min_periods=25).std()
# Mean-reversion deceleration, strengthened when recent participation is elevated:
# ((60d return)/3 - 20d return) / realized vol * log(20d avg volume / 60d avg volume).
particip=(V.rolling(20,min_periods=10).mean()/(V.rolling(60,min_periods=30).mean()+1e-12)).clip(lower=0.05)
f=((P.pct_change(60)/3-P.pct_change(20))/(rv+1e-12))*np.log(particip)
print('rows',len(P),'instruments',len(P.columns),'span',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=q.ic
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,6),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),6))
 print('years',q.groupby(q.index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
