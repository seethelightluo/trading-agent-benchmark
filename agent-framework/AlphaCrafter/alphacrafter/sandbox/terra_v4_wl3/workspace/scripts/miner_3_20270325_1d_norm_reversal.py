import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-25')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.loc[d.index<=cut,'close']
P=pd.DataFrame(px).dropna(how='all'); R=P.pct_change()
# candidate: one-day reversal normalized by 5d volatility, cross-sectional residual
F=-(R.shift(0)).div(R.rolling(5).std())
# at date t, R[t] known, forward return t+1
fr=P.shift(-1).div(P)-1
rows=[]
for dt in P.index:
 x=F.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1), (a.ic>0).mean(), F.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2021-12-31'),('2022','2023-12-31'),('2024','2025-12-31'),('2026','2027-03-25')]:
 q=a.loc[lo:hi,'ic']; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10]:
 fy=P.shift(-h).div(P)-1; vals=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(vals),'IC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1))
# save signal artifact
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20270325_1d_norm_reversal_signal.csv',index=False)
