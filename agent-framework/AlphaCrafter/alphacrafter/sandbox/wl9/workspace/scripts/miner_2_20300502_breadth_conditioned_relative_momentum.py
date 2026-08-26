import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-05-01')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.rename(s) for s in U}
p=pd.concat(D.values(),axis=1).sort_index().loc[:end]
r=p.pct_change(); r20=p.pct_change(20)
breadth=(r>0).mean(axis=1).rolling(10,min_periods=8).mean()
med=r20.median(axis=1).to_numpy()[:,None]
rel=pd.DataFrame(r20.to_numpy()-med,index=r20.index,columns=r20.columns)
sig=rel.mul((2*breadth-1).to_numpy(),axis=0).shift(1)
for h in [1,5,10,20,40,60]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('early',a.loc[:'2023-12-31']),('middle',a.loc['2024-01-01':'2026-12-31']),('late',a.loc['2027-01-01':])]:
  print(' ',name,len(sl),round(sl.ic.mean(),6) if len(sl) else None,round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6) if len(sl)>1 else None)
rank=sig.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)/2).mean()
print('turnover_proxy',round(turn,6))
sig.index.name='date'; sig.to_csv('scripts/miner_2_20300502_breadth_conditioned_relative_momentum_signal.csv')
