import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
prices=pd.DataFrame(px).sort_index()
# Range-efficiency trend: signed 20d move divided by path length, lagged one day.
r=prices.pct_change()
signal=(prices.pct_change(20)/(r.abs().rolling(20).sum()+1e-12)).shift(1)
# evaluate forward close returns
print('range',prices.index.min(),prices.index.max(),'assets',len(px))
for h in [1,5,10,20]:
 fwd=prices.shift(-h)/prices-1
 vals=[]; dates=[]; ns=[]
 for dt in signal.index:
  x=signal.loc[dt]; y=fwd.loc[dt]
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 z=np.array(vals); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),np.mean(z>0)))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<=pd.Timestamp(hi+'-12-31'))]
  print(' ',lo+'-'+hi,len(q), '%.6f %.6f'%(np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12)) if len(q) else '')
# turnover 10-day rank signal
ranks=signal.rank(axis=1,pct=True); a=ranks.diff(10).abs().mean(axis=1).dropna(); print('turn10',a.mean(),'coverage',signal.notna().sum().sum()/(signal.shape[0]*signal.shape[1]))
