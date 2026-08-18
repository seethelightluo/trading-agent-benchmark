import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
rows=[]
for s,x in D.items():
 clv=-(2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1); intra=-(x.close/x.open-1); r=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'clv':clv,'intra':intra,'r':r})
 for dt,g in z.groupby(z.index):
  g=g.dropna()
  if len(g)>=8:
   X=np.c_[np.ones(len(g)),g.clv.values]; b=np.linalg.lstsq(X,g.intra.values,rcond=None)[0]; f=g.intra.values-X@b
   rows.append((dt,spearmanr(f,g.r).statistic,spearmanr(g.intra,g.r).statistic,len(g),np.corrcoef(f,g.clv)[0,1]))
a=pd.DataFrame(rows,columns=['date','icres','icraw','n','res_clv_corr']); a.date=pd.to_datetime(a.date); a=a.set_index('date')
for c in ['icres','icraw']: a[c]=a[c].replace([np.inf,-np.inf],np.nan)
print('dates',len(a),'meanN',a.n.mean(),'raw',a.icraw.mean(),a.icraw.mean()/a.icraw.std(ddof=1),'res',a.icres.mean(),a.icres.mean()/a.icres.std(ddof=1),'hit',(a.icres>0).mean(),'corr',a.res_clv_corr.abs().mean())
for y,g in a.groupby(a.index.year): print(y,len(g),round(g.icres.mean(),4),round(g.icres.mean()/g.icres.std(ddof=1),4))
# horizon decay using next h-day close returns, residualized signal
for h in [1,5,10]:
 rows=[]
 for dt in sorted(set.intersection(*[set(x.index) for x in D.values()])):
  vals=[]; rets=[]; cl=[]
  for s,x in D.items():
   if dt not in x.index: continue
   i=x.index.get_loc(dt)
   if i+ h>=len(x): continue
   intra=-(x.close.iloc[i]/x.open.iloc[i]-1); cv=x.close.iloc[i]; rr=x.close.iloc[i+h]/cv-1
   lo=x.low.iloc[i]; hi=x.high.iloc[i]; c=-(2*(cv-lo)/(hi-lo)-1) if hi!=lo else np.nan
   if np.isfinite(intra+c+rr): vals.append((intra,c)); rets.append(rr)
  if len(vals)>=8:
   v=np.array(vals); f=v[:,0]-np.c_[np.ones(len(v)),v[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(v)),v[:,1]],v[:,0],rcond=None)[0]
   rows.append(spearmanr(f,rets).statistic)
 aa=pd.Series(rows).dropna(); print('h',h,'dates',len(aa),'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1))
