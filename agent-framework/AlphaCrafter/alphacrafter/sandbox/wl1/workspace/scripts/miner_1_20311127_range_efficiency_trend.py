import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
data={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 data[s]=d
close=pd.DataFrame({s:d.close for s,d in data.items()}); high=pd.DataFrame({s:d.high for s,d in data.items()}); low=pd.DataFrame({s:d.low for s,d in data.items()})
# range-efficiency trend: lagged 20d return divided by cumulative daily high-low normalized by price
ret=close.pct_change()
tr=(high-low)/close.shift(1)
raw=close.pct_change(20)/(tr.rolling(20).mean()+1e-8)
sig=raw.shift(1)
res={h:[] for h in [1,5,10,20]}; dates={h:[] for h in res}
for dt in sig.index:
 for h in res:
  if dt not in close.index: continue
  f=sig.loc[dt]; fr=close.shift(-h).loc[dt]/close.loc[dt]-1 if dt in close.index else None
  z=pd.concat([f,fr],axis=1).dropna()
  if len(z)>=8:
   res[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates[h].append(dt)
print('dates/instruments',len(sig),len(close.columns), 'valid', {h:len(v) for h,v in res.items()})
for h,v in res.items():
 a=np.array(v); print(h,'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0)))
# regimes 3-year-ish
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 m=(pd.Series(dates[10]).dt.strftime('%Y')>=lo)&(pd.Series(dates[10]).dt.strftime('%Y')<=hi)
 a=np.array(res[10])[m.values]; print(lo,hi,len(a),np.mean(a) if len(a) else np.nan, np.mean(a)/(np.std(a,ddof=1)+1e-12) if len(a)>1 else np.nan)
# coverage and turnover rank
print('coverage',sig.notna().sum().sum()/sig.size)
r=sig.rank(axis=1,pct=True); print('turnover',np.nanmean((r-r.shift(1)).abs().mean(axis=1)))
