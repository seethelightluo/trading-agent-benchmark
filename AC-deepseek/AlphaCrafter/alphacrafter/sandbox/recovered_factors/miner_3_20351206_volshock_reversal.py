import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 D[a]=x
# align prices by date; factor only lagged information
px=pd.DataFrame({a:D[a]['close'] for a in assets})
ret=px.pct_change()
# 10d reversal, gated by volatility shock: elevated recent vol relative to slow vol
r10=px.pct_change(10).shift(1)
vol5=ret.rolling(5).std().shift(1)
vol60=ret.rolling(60).std().shift(1)
shock=(vol5/(vol60+1e-12)-1).clip(-3,3)
f=(-r10*shock).replace([np.inf,-np.inf],np.nan)
# forward returns
outs={h:[] for h in [1,5,10,20]}; dates=[]; breadth=[]
for dt in px.index:
 vals=f.loc[dt]
 if dt not in px.index: continue
 row=[]
 for a in assets:
  if pd.notna(vals[a]): row.append(a)
 if len(row)<8: continue
 dates.append(dt); breadth.append(len(row))
 for h in outs:
  fr=px.shift(-h).loc[dt]/px.loc[dt]-1
  z=pd.DataFrame({'f':vals[row],'r':fr[row]}).dropna()
  outs[h].append(spearmanr(z.f,z.r).statistic if len(z)>=8 else np.nan)
print('idea=10d reversal x volatility shock; dates',len(dates),'mean breadth',np.mean(breadth),'coverage',f.notna().mean().mean())
for h,v in outs.items():
 v=np.array(v); v=v[np.isfinite(v)]
 print('H',h,'IC %.5f ICIR %.5f hit %.3f n %d'%(v.mean(),v.mean()/(v.std(ddof=1)+1e-12),np.mean(v>0),len(v)))
 for lo,hi in [('2026-01-01','2030-12-31'),('2031-01-01','2035-12-31')]:
  q=np.array([v for d,v in zip(dates,outs[h]) if lo<=str(d.date())<=hi and np.isfinite(v)])
  print(lo, 'n',len(q),'IC %.5f ICIR %.5f'%(q.mean() if len(q) else np.nan,(q.mean()/(q.std(ddof=1)+1e-12)) if len(q)>1 else np.nan))
# rank turnover
ranks=f.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).mean()
print('rank_turnover',turn)
