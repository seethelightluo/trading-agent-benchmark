import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
paths=['../persistent/index_data','../persistent/stock_data']
px={}
for s in U+['DXY']:
 for b in paths:
  try: d=pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']); break
  except FileNotFoundError: d=None
 if d is None: print('missing',s); continue
 px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); dxy=R['DXY']
for h in [1,5,10]:
 byasset={}
 for s in U:
  r=R[s]; beta=r.rolling(60,min_periods=45).cov(dxy)/dxy.rolling(60,min_periods=45).var()
  f=r.rolling(20,min_periods=20).sum()-beta*dxy.rolling(20,min_periods=20).sum()
  y=r.shift(-h).rolling(h,min_periods=h).sum(); byasset[s]=pd.DataFrame({'f':f,'y':y})
 vals=[]; ns=[]; ranks=[]
 for dt in P.index:
  a=[(s,byasset[s].loc[dt,'f'],byasset[s].loc[dt,'y']) for s in U if s in byasset and dt in byasset[s].index and np.isfinite(byasset[s].loc[dt]).all()]
  if len(a)>=8:
   ic=spearmanr([x[1] for x in a],[x[2] for x in a]).statistic
   if np.isfinite(ic): vals.append(ic);ns.append(len(a))
   if h==1:ranks.append(pd.Series({x[0]:x[1] for x in a}).rank(pct=True))
 print('h',h,'dates',len(vals),'avg_names',round(np.mean(ns),2),'IC',round(np.mean(vals),6),'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6),'hit',round(np.mean(np.array(vals)>0),4))
if ranks:
 q=pd.DataFrame(ranks); print('coverage',round(q.notna().mean().mean(),4),'rank_turnover',round(q.diff().abs().mean().mean(),4))
print('dates',len(P),'assets',len(U),'loaded',len(px))
