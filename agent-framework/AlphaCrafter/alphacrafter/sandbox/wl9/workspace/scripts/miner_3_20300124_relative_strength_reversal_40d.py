import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-01-24')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r10=p.pct_change(10)
# Relative losers mean-revert over a medium horizon; shift one completed session.
f=-(r10.sub(r10.median(axis=1),axis=0)); sig=f.shift(1)
y=p.shift(-40)/p-1
rows=[]
for dt in p.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((dt,q,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('range',p.index.min(),p.index.max(),'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4))
print('40d IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
for name,sl in [('early',a.loc[:'2023-12-31']),('middle',a.loc['2024-01-01':'2026-12-31']),('late',a.loc['2027-01-01':])]:
 print(name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
# decay for same lagged signal
for h in [10,20,40,60]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q)
 v=np.array(vals); print('decay',h,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/(v.std(ddof=1)+1e-12),6))
out=f; out.index.name='date'; out.to_csv('scripts/miner_3_20300124_relative_strength_reversal_40d_signal.csv')
