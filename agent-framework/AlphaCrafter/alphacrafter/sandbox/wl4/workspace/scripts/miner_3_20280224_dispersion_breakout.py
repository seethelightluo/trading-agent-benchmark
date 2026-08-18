import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
d={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): d[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
D=pd.concat({a:x for a,x in d.items()},axis=1).sort_index(); C=D.xs('close',axis=1,level=1); H=D.xs('high',axis=1,level=1); L=D.xs('low',axis=1,level=1)
r=C.pct_change(); disp=r.rolling(20).std().mean(axis=1)-r.rolling(20).mean().std(axis=1); z=(disp-disp.rolling(60).mean())/disp.rolling(60).std()
clv=((C-L)/(H-L).replace(0,np.nan)-.5).shift(1); f=clv.mul((1+z.shift(1).clip(-1,2)),axis=0)
ics=[]; prev=None; turns=[]; ni=[]
for t in f.index:
 y=C.shift(-5).loc[t]/C.loc[t]-1; x=f.loc[t]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(q): ics.append(q);ni.append(ok.sum())
  rr=x[ok].rank(pct=True)
  if prev is not None: turns.append((rr-prev.reindex(rr.index)).abs().mean())
  prev=rr
s=pd.Series(ics); print('dates',len(s),'avg_instruments',np.mean(ni),'coverage_pct',100*len(s)/(len(f.index)-5));print('IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'turnover',np.mean(turns))
for h in [1,5,10,20]:
 y=C.shift(-h)/C-1;q=[]
 for t in f.index:
  ok=f.loc[t].notna()&y.loc[t].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[t][ok],y.loc[t][ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
