import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<='2026-07-15')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
r=C.pct_change(); clv=(2*(C-L)/(H-L).replace(0,np.nan)-1)
rec=(C/C.rolling(60,min_periods=40).min()-1); vol=r.rolling(20,min_periods=15).std()
F=(0.55*clv.rolling(5,min_periods=3).mean()/vol + 0.45*rec/vol).rank(axis=1,pct=True).shift(1); Y=r.shift(-1)
q=[]; ns=[]; ts=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8: q.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ts.append(dt)
q=np.asarray(q); ts=np.asarray(ts,dtype='datetime64[ns]')
print('candidate clv5_recovery60_volnorm','dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:
 x=q[-k:]; print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'n',len(x))
for h in [1,5,10]:
 yy=C.pct_change(h).shift(-h); qq=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':yy.loc[dt]}).dropna()
  if len(z)>=8: qq.append(spearmanr(z.f,z.y).statistic)
 qq=np.asarray(qq); print('horizon',h,'IC',round(qq.mean(),6),'ICIR',round(qq.mean()/qq.std(ddof=1),6))
for a,b in [('2020','2021'),('2022','2022'),('2023','2024'),('2025','2026')]:
 m=(ts>=np.datetime64(a+'-01-01'))&(ts<=np.datetime64(b+'-12-31')); print('regime',a,b,'IC',round(q[m].mean(),6),'n',int(m.sum()))
