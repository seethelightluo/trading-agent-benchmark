import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:'2035-04-11']
r=P.pct_change(); vol=r.rolling(20).std()*np.sqrt(10)
# One idea: breadth-conditioned medium-term trend. 10d volatility-scaled momentum
# is activated by the cross-asset fraction with positive 5d returns; all inputs lagged.
breadth=(r.rolling(5).sum()>0).mean(axis=1)
F=(r.rolling(10).sum().div(vol) * (0.5+breadth)).shift(1)
print('rows',len(P),'assets',len(A),'cutoff',P.index[-1].date(),'valid_cells',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 fw=P.shift(-h).div(P)-1; q=[]; ds=[]; nn=[]
 for t in F.index:
  z=pd.concat([F.loc[t],fw.loc[t]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(t);nn.append(len(z))
 q=pd.Series(q,index=ds); print('H',h,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f N %.2f'%(q.mean(),q.mean()/q.std(),(q>0).mean(),np.mean(nn)))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-04-11')]:
  x=q.loc[lo:hi]
  if len(x): print(' regime',lo,'n',len(x),'IC %.6f ICIR %.6f hit %.4f'%(len(x),x.mean(),x.mean()/x.std(),(x>0).mean()))
 for end in ['2034-04-11','2035-04-11']:
  x=q.loc[:end].tail(252)
  if len(x):print(' recent',end,'n',len(x),'IC %.6f ICIR %.6f'%(len(x),x.mean(),x.mean()/x.std()))
 print(' rank_turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('AUDIT max_abs_library_correlation=unavailable: admitted factor signal panels are not persisted; candidate fails mandatory correlation evidence')
