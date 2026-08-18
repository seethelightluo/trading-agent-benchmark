import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-23')
ps={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); d=d[d.date<=end].sort_values('date').set_index('date'); ps[s]=d.close.astype(float)
p=pd.DataFrame(ps).sort_index(); r=p.pct_change(20); # idiosyncratic relative strength vs contemporaneous cross-section
f=r.sub(r.median(axis=1),axis=0); y=p.shift(-10)/p-1
ics=[]; inst=[]; tv=[]; last=None
for dt in f.index:
 x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],z[ok]).statistic);inst.append(ok.sum())
  ranks=x.rank(pct=True); tv.append(np.nan if last is None else np.mean(abs(ranks-last)));last=ranks
A=np.array(ics); print('20d_relative_strength',len(A),np.mean(inst),np.mean(inst)/15,np.mean(A),np.mean(A)/np.std(A,ddof=1)*np.sqrt(252),np.mean(A>0),np.nanmean(tv),f'{f.index.min().date()} {f.index.max().date()}')
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[dt][ok],yy.loc[dt][ok]).statistic)
 q=np.array(q);print('decay',h,round(np.mean(q),6),round(np.mean(q)/np.std(q,ddof=1)*np.sqrt(252),5))
