import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
r=p.pct_change()
# Curvature: recent 20d performance relative to average 60d pace; contrarian signal.
# Vol scaling and one-day lag ensure only completed history is used.
curv=r.rolling(20).sum()-r.rolling(60).sum()/3
vol=r.rolling(60).std()*np.sqrt(60)+0.05
sig=(-curv/vol).shift(1)
fwd=p.shift(-10)/p-1
rows=[]; turns=[]; dates=[]
for i,d in enumerate(p.index):
 x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append(spearmanr(x[ok],y[ok]).statistic); dates.append(d)
  if i>=10:
   q=sig.iloc[i-10]; oo=q.notna()&x.notna()
   if oo.sum()>=8: turns.append(np.mean(np.abs(x[oo].rank()-q[oo].rank()))/oo.sum())
a=np.array(rows); ds=pd.DatetimeIndex(dates)
print('factor relative_curvature_reversal_20_60')
print('dates',len(a),'avgN',np.mean([((sig.loc[d].notna())&(fwd.loc[d].notna())).sum() for d in ds]),'minN',min([((sig.loc[d].notna())&(fwd.loc[d].notna())).sum() for d in ds]),'coverage',np.mean([((sig.loc[d].notna())&(fwd.loc[d].notna())).sum()/15 for d in ds]))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
for h in [5,10,20,40,60]:
 yy=p.shift(-h)/p-1; z=[]
 for d in p.index:
  x=sig.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.mean(z),len(z))
for lo,hi in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 z=[]
 for d in p.index:
  if lo<=d.year<=hi:
   x=sig.loc[d];y=fwd.loc[d];ok=x.notna()&y.notna()
   if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 print('regime',lo,hi,np.mean(z) if z else None,len(z))
pd.DataFrame(sig,columns=assets).rename_axis('date').to_csv('scripts/miner_1_20351122_relative_curvature_reversal_signal.csv')
