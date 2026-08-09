"""One candidate: residual-downside recovery dispersion, cutoff 2032-05-12.
Cross-asset signal: persistence of dispersion in idiosyncratic next-day repairs after downside events.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
START=pd.Timestamp('2026-07-16'); END=pd.Timestamp('2032-05-12')
def col(a,x):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[START:END,x].astype(float)
c=pd.DataFrame({a:col(a,'close') for a in A}); o=pd.DataFrame({a:col(a,'open') for a in A})
r=c.pct_change(fill_method=None); market=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(market).div(market.rolling(60,min_periods=45).var(),axis=0); resid=r-beta.mul(market,axis=0)
intra=c.div(o).sub(1); scale=intra.rolling(60,min_periods=45).std()
# On a prior residual-downside event, normalize the ensuing intraday repair.
# Factor is 20d vs 60d widening in its asset-specific repair variability: assets
# with a recently more discriminating repair process are hypothesized to retain recovery optionality.
repair=(intra.div(scale).clip(-5,5)).where(resid.shift(1)<0)
short=repair.rolling(20,min_periods=6).std(); long=repair.rolling(60,min_periods=15).std()
f=short.div(long).sub(1)
print('FACTOR residual_downside_repair_dispersion_persistence_20_60obs cutoff',END.date(),'assets',len(A))
print('formula: std_20[clip((C/O-1)/sd_60(C/O-1),-5,5) | residual_return(t-1)<0] / std_60[same] - 1')
print('coverage',int(f.notna().sum().sum()),'/',f.size,round(f.notna().mean().mean(),6))
OUT={}
for h in (1,5,10,20):
 y=c.shift(-h).div(c).sub(1); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z); ds=pd.DatetimeIndex(ds); OUT[h]=(z,ds)
 print('H',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'min_n',min(ns))
for label,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_cutoff','2030-01-01','2032-05-12'),('recent_12m','2031-05-13','2032-05-12')]:
 z,d=OUT[20]; x=z[(d>=lo)&(d<=hi)]; print('REGIME',label,'H20 dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True); print('TURNOVER',round((rnk-rnk.shift()).abs().stack().mean(),6),'median_iqr',round((f.quantile(.75,axis=1)-f.quantile(.25,axis=1)).median(),6))
print('NOVELTY pending: candidate is not admission-eligible without exact full current-library signal correlation audit.')
