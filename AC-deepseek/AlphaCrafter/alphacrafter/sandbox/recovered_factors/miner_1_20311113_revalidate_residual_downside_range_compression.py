"""Miner-1 revalidation: residual downside range compression persistence, completed bars through 2031-11-12."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-11-12'); START=pd.Timestamp('2026-07-16')
def col(a,c):
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,c].astype(float)
close=pd.DataFrame({a:col(a,'close') for a in A}); high=pd.DataFrame({a:col(a,'high') for a in A}); low=pd.DataFrame({a:col(a,'low') for a in A})
r=close.pct_change(fill_method=None); med=r.median(axis=1); beta=pd.DataFrame({a:r[a].rolling(60,min_periods=45).cov(med)/med.rolling(60,min_periods=45).var() for a in A}); res=r-beta.mul(med,axis=0)
rng=(high-low).div(close.replace(0,np.nan)); down=res.shift(1)<0
# Conditional event averages: min events 5 / 12; high means recent residual-downside ranges compressed vs baseline.
def condmean(w,minimum):
 num=rng.where(down).rolling(w,min_periods=1).sum(); den=down.astype(float).rolling(w,min_periods=1).sum()
 return (num/den).where(den>=minimum)
f=-np.log(condmean(20,5)/condmean(60,12))
print('FACTOR residual_downside_range_compression_persistence_20_60obs','cutoff',END.date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=close.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index[(f.index>=START)&(f.index<=END-pd.Timedelta(days=0))]:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(t); ns.append(len(q))
 z=np.array(z); ds=pd.DatetimeIndex(ds); R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),5),'mean_n',round(np.mean(ns),2),'min_n',min(ns),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=20; z,ds,_=R[best]
for name,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_2031','2030-01-01',END)]:
 q=z[(ds>=lo)&(ds<=hi)]; print('REGIME',name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),5))
rnk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER',round(float(np.mean(turns)),6),'comparisons',len(turns),'CONCENTRATION_MEDIAN_IQR',round(float(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median()),6))
