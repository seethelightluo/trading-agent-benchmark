import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-05-12')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d[d.index<=end]
p=pd.DataFrame(px).sort_index()
r=p.pct_change(5)
# signal observed at t: use prices through t-1, hence shifted 1; residual relative to cross-sectional mean
sig=-(r.sub(r.mean(axis=1),axis=0)).shift(1)
fwd=p.pct_change(1).shift(-1)
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean(),(sig.rank(axis=1).diff().abs().mean(axis=1).dropna()/14).mean()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic; print(lo, len(q), q.mean(),q.mean()/q.std())
for h in [1,5,10,20]:
 fy=p.pct_change(h).shift(-h); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.mean(rr),'dates',len(rr))
sig.to_csv('scripts/miner_2_20270513_residual_reversal_signal.csv')
