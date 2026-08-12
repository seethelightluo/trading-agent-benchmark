import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
dates=sorted(set.intersection(*[set(s.index) for s in px.values()]))
D=pd.DataFrame({a:px[a] for a in assets}).reindex(dates).sort_index()
# short-horizon contrarian signal, normalized by recent risk; all inputs lagged via shift(1)
r=D.pct_change(1); r5=D.pct_change(5); vol=D.pct_change().rolling(20).std()
sig=(-(r5/vol.replace(0,np.nan))).shift(1)
rows=[]
for h in [1,5,10,20]:
  fwd=D.pct_change(h).shift(-h)
  ics=[]; turns=[]; ns=[]
  for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
      ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
  q=pd.Series(ics).dropna(); mean=q.mean(); sd=q.std(ddof=1)
  print(h, 'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(mean,6),'ICIR',round(mean/sd*np.sqrt(252),6),'hit',round((q>0).mean(),4))
# turnover based rank ordering
rank=sig.rank(axis=1,pct=True); print('coverage',round(sig.notna().sum().sum()/sig.size,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),5))
for yr in [2024,2025,2026,2027,2028,2029,2030]:
 q=[]
 fwd=D.pct_change(1).shift(-1)
 for dt in sig.index:
  if dt.year!=yr: continue
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic)
 q=pd.Series(q).dropna(); print('yr',yr,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4) if len(q)>1 else None)
sig.to_csv('scripts/miner_1_20301114_short_reversal_signal.csv')
