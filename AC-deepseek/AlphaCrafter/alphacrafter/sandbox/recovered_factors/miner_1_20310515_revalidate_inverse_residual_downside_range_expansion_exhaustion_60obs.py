"""Revalidate one admitted idea: inverse residual-downside normalized range exhaustion."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2031-05-14');W=60
def ld(a,c='close'):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
def bet(x,y,cond=None,mp=45):
 o={}
 for a in A:
  xx=x[a]; yy=y
  if cond is not None: xx=xx.where(cond);yy=yy.where(cond)
  o[a]=xx.rolling(W,min_periods=mp).cov(yy)/yy.rolling(W,min_periods=mp).var()
 return pd.DataFrame(o)
p=pd.DataFrame({a:ld(a) for a in A});hi=pd.DataFrame({a:ld(a,'high') for a in A});lo=pd.DataFrame({a:ld(a,'low') for a in A})
r=p.pct_change(fill_method=None);m=r.median(axis=1);b=bet(r,m);res=r-b.mul(m,axis=0)
rng=(hi-lo).abs()/p.shift(); norm=rng/rng.rolling(20,min_periods=15).median().replace(0,np.nan)
f=pd.DataFrame({a:-norm[a].where(res[a].shift()<0).rolling(W,min_periods=12).mean() for a in A})
print('FACTOR inverse_residual_downside_range_expansion_exhaustion_60obs','cutoff',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H);z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds); ir=z.mean()/z.std(ddof=1) if len(z)>1 and z.std(ddof=1)>0 else np.nan;R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(ir,6),'hit',round((z>0).mean(),4),'mean_n',round(np.mean(ns),2),'PASS',abs(z.mean())>=.007 and abs(ir)>=.084)
best=max(R,key=lambda h:abs(R[h][0].mean()*(R[h][0].mean()/R[h][0].std(ddof=1))) if len(R[h][0])>1 and R[h][0].std(ddof=1)>0 else -1);z,ds,_=R[best];print('SELECTED',best)
for n,x,y in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_current','2026-01-01',END)]:
 q=z[(ds>=x)&(ds<=y)];print('REGIME',n,'dates',len(q),'IC',round(q.mean(),6) if len(q) else 'NA','ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 and q.std(ddof=1)>0 else 'NA','hit',round((q>0).mean(),4) if len(q) else 'NA')
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER',round(np.mean(turn),6) if turn else 'NA','comparisons',len(turn),'CONCENTRATION_MEAN_SD',round(f.std(axis=1).mean(),6))
# Signal-level novelty evidence from every active JSON, using definitions unavailable generically is assessed from persisted signal audit pkl when present.
import os,json
files=[x for x in os.listdir('factors') if x.endswith('.json') and '_deprecated' not in x]
print('LIBRARY_FILES',len(files))
# Existing historical audit established 30 comparisons, but revalidation must flag no newly calculable signal evidence.
print('MAX_ABS_LIBRARY_CORRELATION','NOT_RECOMPUTED','reason','factor JSON contains definitions but not signal panels; no generic signal reconstruction is valid')
