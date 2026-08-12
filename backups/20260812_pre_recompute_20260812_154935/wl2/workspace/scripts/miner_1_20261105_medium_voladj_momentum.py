import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2026-11-04')
def load(s):
 return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index().loc[:CUT]
PX={s:load(s) for s in U}; R=pd.concat({s:x.pct_change() for s,x in PX.items()},axis=1).sort_index()
# Novel medium-horizon signal: 20d return scaled by 60d realized volatility, lagged one session.
F=pd.concat({s:(x.pct_change(20)/(x.pct_change().rolling(60).std()*np.sqrt(20)+.01)).shift(1) for s,x in PX.items()},axis=1).sort_index()
rows=[]
for dt in F.index:
 vals=[]
 for s in U:
  if pd.notna(F.at[dt,s]):
   aft=R[s].loc[R[s].index>dt]
   if len(aft): vals.append((F.at[dt,s],aft.iloc[0]))
 z=pd.DataFrame(vals).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z[0],z[1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=lagged 20d return/(60d vol*sqrt20+.01), cutoff',CUT.date())
print('dates',len(d),'avgN',round(d.n.mean(),2),'IC',round(d.ic.mean(),6),'ICIR',round(d.ic.mean()/d.ic.std(ddof=1),6),'hit',round((d.ic>0).mean(),4),'coverage',round(d.n.sum()/(len(d)*15),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex(d.index).mean(),4))
for name,g in d.groupby(d.index.year): print(name,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
for h in [3,5,10]:
 rr=[]
 for dt in F.index:
  vals=[]
  for s in U:
   if pd.notna(F.at[dt,s]):
    aft=R[s].loc[R[s].index>dt].iloc[:h]
    if len(aft)==h: vals.append((F.at[dt,s],(1+aft).prod()-1))
  z=pd.DataFrame(vals).dropna()
  if len(z)>=8: rr.append(spearmanr(z[0],z[1]).statistic)
 print('decay',h,'days',len(rr),'IC',round(np.mean(rr),6),'ICIR',round(np.mean(rr)/np.std(rr,ddof=1),6))
print('last usable',max(x.index.max() for x in PX.values()).date())
