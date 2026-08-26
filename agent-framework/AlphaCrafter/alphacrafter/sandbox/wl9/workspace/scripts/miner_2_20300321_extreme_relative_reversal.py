import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-03-21')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
# Nonlinear extreme relative reversal: emphasize tails of 20d cross-sectional relative returns,
# while retaining direction; lag one completed session.
r20=p.pct_change(20)
rel=r20.sub(r20.median(axis=1),axis=0)
rk=rel.rank(axis=1,pct=True)
# centered rank reversal, magnitude-weighted to prioritize unusually large relative moves
f=-(rk-0.5)*(2*abs(rk-0.5)).pow(0.75)
f=f.rolling(3,min_periods=2).mean().shift(1)
rows=[]
for h in [10,20,40,60]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n'])
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 if h==40:
  for name,sl in [('early',a[a.date<='2023-12-31']),('middle',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('late',a[a.date>='2027-01-01'])]:
   print(name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
x=f.rank(axis=1,pct=True); turnover=(x.diff().abs().mean(axis=1)/2).dropna()
print('turnover_proxy',round(turnover.mean(),6),'signal_dates',int(f.notna().any(axis=1).sum()))
f.index.name='date'; f.to_csv('scripts/miner_2_20300321_extreme_relative_reversal_signal.csv')
