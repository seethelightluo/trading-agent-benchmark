import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<='2026-07-15')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U}); V=pd.DataFrame({s:D[s].volume.reindex(dates) for s in U})
# Volume-confirmed close-location trend: persistent buying pressure, scaled by unusual volume.
clv=(2*C-H-L)/(H-L).replace(0,np.nan)
volsur=V/V.rolling(20,min_periods=10).median()-1
F=(clv.rolling(5,min_periods=3).mean()* (1+volsur.clip(-.8,2))).rank(axis=1,pct=True)
F=F.shift(1); Y=C.pct_change().shift(-1)
q=[]; ns=[]; turns=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 turns.append(F.loc[dt].rank(pct=True))
q=np.asarray(q); print('dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(pd.DataFrame(turns).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:
 x=q[-k:]; print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,3,5]:
 yy=C.pct_change(h).shift(-h)
 qq=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':yy.loc[dt]}).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic)
 print('horizon',h,'IC',round(np.mean(qq),6),'ICIR',round(np.mean(qq)/np.std(qq,ddof=1),6),'n',len(qq))
