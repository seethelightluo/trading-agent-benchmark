import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}; p=pd.DataFrame(px).sort_index().astype(float)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
q=.8; raw=v>v.rolling(120,min_periods=60).quantile(q); gate=(raw.shift(1)&raw.shift(2)).astype(bool)
sig=(-(p/p.shift(20)-1)).where(gate); fr=p.shift(-10)/p-1
vals=[]; dates=[]; ns=[]
for d in p.index:
 z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(d);ns.append(len(z))
a=np.asarray(vals); dates=pd.DatetimeIndex(dates)
print('candidate=lagged_vix_q80_anti_momentum_20obs horizon=10')
print('dates',len(a),'instruments',len(p.columns),'meanN',np.mean(ns),'coverage_cells',sig.notna().sum().sum()/sig.size,'gate_fraction',gate.mean())
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; x=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.asarray(x);print('decay',h,len(x),x.mean(),x.mean()/x.std(ddof=1))
# turnover proxy: daily rank changes among active dates
r=sig.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).dropna();print('turnover_proxy',turn.mean())
# available comparator audit (not full library; therefore admission evidence remains incomplete)
comparators={'raw_anti_mom20':-(p/p.shift(20)-1),'raw_mom20':p/p.shift(20)-1,'anti_mom10':-(p/p.shift(10)-1)}
for k,c in comparators.items():
 z=pd.concat([sig.stack().rename('s'),c.stack().rename('c')],axis=1).dropna();print('corr',k,spearmanr(z.s,z.c).statistic,'cells',len(z))
