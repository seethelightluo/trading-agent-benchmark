import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d
p=pd.DataFrame(px).sort_index()
r=np.log(p).diff()
# candidate: medium-term return per realized volatility, lagged one completed day
sig=(r.rolling(20,min_periods=15).sum()/r.rolling(20,min_periods=15).std()).shift(1)
# forward close-to-close returns from signal date
out=[]
for h in [1,5,10,20]:
 f=np.log(p.shift(-h)/p)
 ics=[]; dates=[]; ns=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=f.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(ics); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a),np.mean(ns)))
# regime daily
f=np.log(p.shift(-1)/p); arr=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8: arr.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(arr,columns=['date','ic']).set_index('date')
for lo,hi in [('2024','2027'),('2028','2030'),('2031','2034')]:
 q=a.loc[lo:hi,'ic']; print('REG',lo,hi,'IC %.6f ICIR %.6f dates %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',sig.notna().sum().sum()/sig.size)
# turnover of cross-sectional ranks sampled every 10 days
ranks=sig.rank(axis=1,pct=True); vals=[]
for i in range(10,len(ranks),10):
 x=ranks.iloc[i-10];y=ranks.iloc[i]; z=pd.concat([x,y],axis=1).dropna(); vals.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover_proxy',np.mean(vals))
