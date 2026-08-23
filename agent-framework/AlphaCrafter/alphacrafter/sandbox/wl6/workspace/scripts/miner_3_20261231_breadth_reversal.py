import pandas as pd, numpy as np, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-30'); base='../persistent/stock_data'
px={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); px[a]=d.sort_values('date').set_index('date').close.loc[:cut]
P=pd.DataFrame(px); R=P.pct_change(); breadth=R.gt(0).sum(axis=1)/R.notna().sum(axis=1)
# lagged, extreme market breadth; factor rewards reversal after broad moves, otherwise zero
ext=((breadth-0.5).abs().rolling(60,min_periods=30).rank(pct=True).shift(1).clip(lower=.7)-.7).clip(lower=0)/.3
market_move=R.mean(axis=1).shift(1)
# directionally stronger reversal after broad one-sided move, volatility normalized
sig={}
for a in A:
 vol=R[a].rolling(20,min_periods=15).std().shift(1)
 sig[a]=((-R[a].shift(1)/vol)*ext*market_move.abs()).replace([np.inf,-np.inf],np.nan)
rows=[]
for a in A:
 f=sig[a]; fr={h:P[a].shift(-h)/P[a]-1 for h in [1,3,5,10]}
 for dt in f.index:
  for h in [1,3,5,10]: rows.append((dt,a,h,f.loc[dt],fr[h].loc[dt]))
dd=pd.DataFrame(rows,columns=['date','a','h','f','r']).dropna()
for h in [1,3,5,10]:
 z=dd[dd.h==h]; obs=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   x=spearmanr(g.f,g.r).statistic
   if np.isfinite(x): obs.append(x)
 obs=np.array(obs); print('H',h,'dates',len(obs),'avg_n',round(z.groupby('date').size().reindex(dd[dd.h==h].date.unique()).dropna().mean(),2),'IC',round(obs.mean(),5),'ICIR',round(obs.mean()/obs.std(ddof=1),5),'hit',round((obs>0).mean(),4))
if True:
 z=dd[dd.h==1]; obs=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: obs.append((dt,spearmanr(g.f,g.r).statistic))
 z=pd.DataFrame(obs,columns=['date','ic']); print('PERIOD',z.date.min().date(),z.date.max().date())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
  q=z[(z.date.dt.year>=int(lo))&(z.date.dt.year<=int(hi))]; print('REG',lo,hi,len(q),round(q.ic.mean(),5),round(q.ic.mean()/q.ic.std(ddof=1),5))
