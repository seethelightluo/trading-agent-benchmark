import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p=f'../persistent/stock_data/{a}.csv'; x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[a]=x.set_index('date')
close=pd.DataFrame({a:D[a].close for a in assets}); op=pd.DataFrame({a:D[a].open for a in assets}); hi=pd.DataFrame({a:D[a].high for a in assets}); lo=pd.DataFrame({a:D[a].low for a in assets})
# candidate: reversal amplified only on high cross-sectional dispersion, lagged one day
r=close.pct_change(); disp=r.rolling(20).std(axis=1).shift(1)
# use cross-sectional rank dispersion; factor is contrarian 5d return times dispersion rank
rev=-close.pct_change(5).shift(1)
csdisp=disp.rank(pct=True,axis=1)
f=rev*csdisp
# forward returns from t to t+h, signal at t
out=[]
for h in [1,5,10,20]:
 fr=close.shift(-h)/close-1
 vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates).dropna(); ic=s.mean(); ir=ic/s.std(ddof=1)*np.sqrt(len(s)) if len(s)>1 else np.nan
 print('H',h,'dates',len(s),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(ic,ir,(s>0).mean()))
 for label,mask in [('2020-24',(s.index<'2025-01-01')),('2025-27',((s.index>='2025-01-01')&(s.index<'2028-01-01'))),('2028-30',(s.index>='2028-01-01')) ,('latest120',s.index>=s.index.max()-pd.Timedelta(days=180))]:
  q=s[mask]; print(label,len(q),round(q.mean(),6) if len(q) else None,round(q.std(ddof=1) and q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6) if len(q)>1 else None)
# coverage turnover
print('coverage',f.notna().sum().sum()/f.size,'turnover',np.nanmean(np.abs(f.rank(pct=True,axis=1).diff(10))))
# proxy correlations with standard signals
for name,x in [('rev5',rev),('ret20',close.pct_change(20).shift(1)),('vol20',r.rolling(20).std().shift(1).mean(axis=1).to_frame().reindex(columns=assets))]:
 a=f.stack(); b=x.stack(); z=pd.concat([a,b],axis=1).dropna();print('rho',name,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))
print('cutoff',close.index.max().date())
