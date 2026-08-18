import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
xs={}
for a in assets:
 f=f'{base}/{a}.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 c=d.close.astype(float); r=c.pct_change()
 # all ingredients shifted one day at signal construction
 mom=c.pct_change(60)
 vol=r.rolling(90,min_periods=45).std()*np.sqrt(60)
 loc=((c-d.low)/(d.high-d.low).replace(0,np.nan)).rolling(20,min_periods=10).mean()
 xs[a]=pd.DataFrame({'f':(mom/vol*loc).shift(1),'p':c})
# shared dates
F=pd.DataFrame({a:x.f for a,x in xs.items()}); P=pd.DataFrame({a:x.p for a,x in xs.items()})
res=[]; artifacts=[]
for h in [10,20,30,40]:
 ic=[]; rows=[]
 for dt in F.index:
  if dt not in P.index: continue
  f=F.loc[dt]; fr=P.shift(-h).loc[dt]/P.loc[dt]-1
  z=pd.concat([f,fr],axis=1).dropna(); n=len(z)
  if n>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   ic.append(v); rows.append((dt,n,v))
 s=pd.Series(ic); print('H',h,'dates',len(s),'avgN',np.mean([x[1] for x in rows]),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
 if h==30:
  for dt,n,v in rows: artifacts.append((dt,v))
# coverage and turnover based rank changes
valid=F.notna().sum(axis=1); print('coverage',valid.mean()/len(assets),'avg valid',valid.mean())
rank=F.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
out=pd.DataFrame(artifacts,columns=['date','ic']); out.to_csv('scripts/artifacts/miner_3_20330317_range_responsive_trend_ic.csv',index=False)
# signal artifact full matrix for audit
F.to_csv('scripts/artifacts/miner_3_20330317_range_responsive_trend_signal.csv')
