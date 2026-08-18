import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# trend consistency: signed 20d return scaled by fraction of positive daily returns, with volatility penalty
ret=P/P.shift(20)-1
pos=r.rolling(20).mean()
vol=r.rolling(20).std()
f=(ret*(0.5+pos)/vol.replace(0,np.nan)).shift(1)
# forward 10d return
fr=P.shift(-10)/P-1
rows=[]
for dt in P.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  rows.append((dt,ic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit', (z.ic>0).mean())
for h in [1,3,5,10,20]:
 fy=P.shift(-h)/P-1; vals=[]
 for dt in P.index:
  ok=f.loc[dt].notna()&fy.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt][ok],fy.loc[dt][ok]).statistic)
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2033')]:
 q=z.loc[lo:hi,'ic']; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std())
# rank turnover
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna(); print('turnover',turn.mean())
# signal artifact
f.to_csv('scripts/miner_2_20331028_trend_consistency_signal.csv')
