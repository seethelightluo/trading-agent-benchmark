import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-10-21')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index().loc[:CUT]
PX={s:load(s) for s in U}; P=pd.concat(PX,axis=1).sort_index(); R=pd.concat({s:x.pct_change() for s,x in PX.items()},axis=1).sort_index(); F=pd.concat({s:(x.pct_change(10)/(x.pct_change().rolling(20).std()*np.sqrt(10)+.01)).shift(1) for s,x in PX.items()},axis=1).sort_index()
# forward return is next observed return per asset, aligned by common date
N=[]; rows=[]
for dt in F.index:
 vals=[]
 for s in U:
  if dt in F[s].index:
   aft=R[s].loc[R[s].index>dt]
   if len(aft): vals.append((F.at[dt,s],aft.iloc[0]))
 z=pd.DataFrame(vals).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z[0],z[1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('candidate=lagged 10d momentum/(20d vol*sqrt10+0.01), cutoff',CUT.date()); print('dates',len(d),'avgN',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(ddof=1),'hit',(d.ic>0).mean(),'coverage',d.n.sum()/(len(d)*15)); print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex(d.index).mean())
for name,g in d.groupby(d.index.year): print(name,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
print('last usable',P.index.max().date())
