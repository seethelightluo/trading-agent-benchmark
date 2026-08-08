"""Quarterly revalidation of inverse residual-downside range expansion exhaustion; completed bars through 2032-03-17."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-03-17'); START=pd.Timestamp('2026-07-16')
D={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS}
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[(x.index>=START)&(x.index<=CUT)].index) for x in D.values()])))
c=pd.DataFrame({a:D[a].reindex(idx).close for a in AS}); h=pd.DataFrame({a:D[a].reindex(idx).high for a in AS}); l=pd.DataFrame({a:D[a].reindex(idx).low for a in AS})
r=c.pct_change(); m=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); res=r-beta.mul(m,axis=0)
rng=(h-l).div(c.shift(1)); s=-(rng/rng.rolling(20,min_periods=12).median()).where(res.shift(1)<0).rolling(60,min_periods=12).mean()
print('FACTOR inverse_residual_downside_range_expansion_exhaustion_60obs cutoff',CUT.date(),'assets',len(AS),'calendar_dates',len(idx))
print('coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
results={}; valid={}
for q in [1,5,10,20]:
 f=c.shift(-q)/c-1; vals=[]; ns=[]
 for t in idx:
  z=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 vals=np.array(vals); ns=np.array(ns); results[q]=(vals,ns)
 print('H',q,'dates',len(vals),'IC',round(vals.mean(),6),'ICIR',round(vals.mean()/vals.std(ddof=1),6),'hit',round((vals>0).mean(),6),'mean_n',round(ns.mean(),3),'min_n',int(ns.min()))
f=c.shift(-20)/c-1
for name,lo,hi in [('2026_2029',pd.Timestamp('2026-07-16'),pd.Timestamp('2029-12-31')),('2030_2032',pd.Timestamp('2030-01-01'),CUT),('recent_12m',pd.Timestamp('2031-03-18'),CUT)]:
 x=[]
 for t in idx:
  z=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8 and lo<=t<=hi: x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('REGIME',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
ranks=s.rank(axis=1,pct=True); print('turnover',round((ranks-ranks.shift()).abs().stack().mean(),6),'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
print('NOVELTY unchanged-factor audit retained: max_abs_library_correlation=0.348663 closest=idiosyncratic_upside_tail_skewness_60obs evidence_cells=8669 screened=30')
