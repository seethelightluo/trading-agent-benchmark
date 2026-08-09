import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for s in U},axis=1).sort_index().loc[:END]
for w in [5,20,30]:
 hi=P.rolling(w,min_periods=max(4,w//2)).max(); lo=P.rolling(w,min_periods=max(4,w//2)).min(); F=-(P-lo)/(hi-lo).replace(0,np.nan); Y=P.shift(-1)/P-1; out=[]
 for d in P.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: out.append((d,z.f.corr(z.y,method='spearman'),len(z)))
 a=pd.DataFrame(out,columns=['d','ic','n']); ic=a.ic
 print(w,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
 print('regimes',[(yr,round(g.mean(),5),round(g.mean()/g.std(ddof=1),5),len(g)) for yr,g in ic.groupby(a.d.dt.year)])
 F.stack().rename('factor').reset_index().to_csv(f'scripts/miner_2_20261217_channel{w}_signal.csv',index=False)
