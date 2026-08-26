import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-07-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# Downside-skew reversal: reverse recent relative performance, but penalize assets
# whose recent downside volatility is large; smooth and lag to avoid lookahead.
med=r.median(axis=1); rel=r.sub(med,axis=0)
down=r.where(r<0).rolling(15,min_periods=8).std()
allvol=r.rolling(15,min_periods=8).std()
# asymmetric risk normalization: downside vol receives extra weight
risk=(0.5*allvol+0.5*down).replace(0,np.nan)
f=(-rel.rolling(15,min_periods=10).sum()/risk).rolling(3,min_periods=2).mean().shift(1)
rows=[]
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']); ic=a.ic.mean(); ir=ic/(a.ic.std(ddof=1)+1e-12)
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 if h==40:
  for name,sl in [('2020-2023',a[(a.date>='2020-01-01')&(a.date<='2023-12-31')]),('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
   print(name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
print('turnover_proxy',round(f.diff().abs().mean().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_2_20300711_downside_skew_reversal_signal.csv')
