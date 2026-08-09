"""Revalidate the single admitted relative-volume participation factor through current cursor."""
import json, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-07-26')
D={a:get_stock_daily_data(a,5000).set_index('date') for a in A}
for x in D.values(): x.index=pd.to_datetime(x.index)
p=pd.DataFrame({a:pd.to_numeric(D[a].loc[:END,'close'],errors='coerce') for a in A})
v=pd.DataFrame({a:pd.to_numeric(D[a].loc[:END,'volume'],errors='coerce') for a in A})
v=v.where(v>0); f=np.log(v/v.rolling(20,min_periods=15).mean())
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'signal_cells',int(f.count().sum()),'of',f.size)
print('NONZERO_VOLUME',json.dumps({a:int(v[a].notna().sum()) for a in A}))
def met(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.factor.corr(z.forward,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(); turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 reg={}
 for n,s in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[s];reg[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if q.std() else np.nan,'hit':(q>0).mean()}
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/sd if sd else np.nan,'hit':(x>0).mean(),'dates':len(x),'se':sd/np.sqrt(len(x)) if len(x) else np.nan,'mean_names':float(np.mean(ns)) if ns else np.nan,'turnover_10d':float(np.mean(turns)) if turns else np.nan,'regimes':reg}
for h in [1,5,10,20]:print('METRIC',json.dumps(met(h),default=float))
# Signal diversity diagnostic: exact values and dates with a usable >=8-name cross section.
counts=f.notna().sum(axis=1); usable=f[counts>=8]
print('USABLE_DATES',len(usable),'MAX_NAMES',int(counts.max()),'UNIQUE_SIGNAL_VALUES',int(pd.unique(f.stack()).size),'CROSS_SECTIONAL_UNIQUE_RANGE',int(usable.nunique(axis=1).min()) if len(usable) else None,int(usable.nunique(axis=1).max()) if len(usable) else None)
print('LATEST_SIGNAL_DATE',f.dropna(how='all').index.max().date() if f.notna().any().any() else None)
