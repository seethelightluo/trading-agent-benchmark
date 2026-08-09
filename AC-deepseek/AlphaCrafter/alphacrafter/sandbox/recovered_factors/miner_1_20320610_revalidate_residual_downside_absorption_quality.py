"""Quarterly revalidation: residual downside absorption quality; completed-bar cutoff 2032-06-09."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-06-09')
def load(a,c):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A}); hi=pd.DataFrame({a:load(a,'high') for a in A}); lo=pd.DataFrame({a:load(a,'low') for a in A})
r=p.pct_change(fill_method=None); med=r.median(axis=1)
b=pd.DataFrame({a:r[a].rolling(60,min_periods=45).cov(med)/med.rolling(60,min_periods=45).var() for a in A})
res=r-b.mul(med,axis=0); sd=res.rolling(60,min_periods=45).std(); severity=(-res/sd).clip(0,4)
clv=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
f=(severity*clv).rolling(20,min_periods=15).sum()/(severity.rolling(20,min_periods=15).sum()+1e-12)-(severity*clv).rolling(60,min_periods=45).sum()/(severity.rolling(60,min_periods=45).sum()+1e-12)
print('FACTOR residual_downside_absorption_quality_20_60obs cutoff',p.index.max().date(),'assets',len(A),'signal cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
OUT={}
for H in [1,5,10,20]:
 y=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z); ds=pd.DatetimeIndex(ds); OUT[H]=(z,ds,ns)
 print('HORIZON',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'min_n',min(ns),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
z,ds,_=OUT[20]
for n,st,en in [('2026_2029','2026-07-16','2029-12-31'),('2030_to_cutoff','2030-01-01',END),('recent_12m','2031-06-10',END)]:
 x=z[(ds>=st)&(ds<=en)];print('REGIME',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
ranks=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER',round(np.mean(turn),6),'comparisons',len(turn),'MEDIAN_IQR',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
print('VALIDATION_RANGE',f.index.min().date(),'to',p.index.max().date())
