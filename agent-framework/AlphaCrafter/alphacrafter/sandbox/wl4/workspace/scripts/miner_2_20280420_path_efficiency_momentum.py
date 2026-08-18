import pandas as pd, numpy as np
from scipy.stats import spearmanr
import os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
    D[s]=x
# common dates and only through requested validation end
end=pd.Timestamp('2028-04-19')
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
dates=[d for d in dates if d<=end]
# Path-efficiency momentum: signed 20d return times directional efficiency (net move / path length)
# signal at t from closes through t; forward return t to t+10
rows=[]
for i,t in enumerate(dates):
    if i+10>=len(dates): break
    vals=[]; fw=[]
    for s in U:
        x=D[s]
        if t not in x.index: continue
        j=x.index.get_loc(t)
        if j<20 or j+10>=len(x.index): continue
        c=x.close.values
        r=c[j]/c[j-20]-1
        path=np.abs(np.diff(c[j-20:j+1])).sum()/c[j-20]
        f=c[j+10]/c[j]-1
        if np.isfinite(r) and path>0 and np.isfinite(f): vals.append(r/path); fw.append(f)
    if len(vals)>=8:
        ic=spearmanr(vals,fw).statistic
        rows.append((t,ic,len(vals)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'min_n',z.n.min(),'coverage',z.n.mean()/15)
for label,q in [('full',z),('early',z.iloc[:len(z)//2]),('late',z.iloc[len(z)//2:]),('recent250',z.tail(250))]:
    print(label,'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
# average rank turnover: rank changes / (n-1)
# recompute cross-sectional signal panel
P=[]
for t in dates:
    row={}
    for s in U:
        x=D[s]; j=x.index.get_indexer([t])[0]
        if j>=20: row[s]= (x.close.iloc[j]/x.close.iloc[j-20]-1)/(np.abs(np.diff(x.close.iloc[j-20:j+1])).sum()/x.close.iloc[j-20])
    if len(row)>=8: P.append(pd.Series(row,name=t).rank(pct=True))
pdct=pd.DataFrame(P); turn=(pdct.diff().abs().mean(axis=1)/2).mean()
print('rank_turnover',turn)
print('decay')
for h in [1,5,10,20]:
    a=[]
    for t in dates:
      vals=[]; fw=[]
      for s in U:
       x=D[s]; j=x.index.get_indexer([t])[0]
       if j>=20 and j+h<len(x):
        sig=(x.close.iloc[j]/x.close.iloc[j-20]-1)/(np.abs(np.diff(x.close.iloc[j-20:j+1])).sum()/x.close.iloc[j-20]); vals.append(sig); fw.append(x.close.iloc[j+h]/x.close.iloc[j]-1)
      if len(vals)>=8:a.append(spearmanr(vals,fw).statistic)
    print(h,np.nanmean(a),len(a))
