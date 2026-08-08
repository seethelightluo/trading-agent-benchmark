import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
I=sorted(set.intersection(*[set(x.index) for x in D.values()]))
o=pd.DataFrame({a:D[a].reindex(I).open for a in A}); c=pd.DataFrame({a:D[a].reindex(I).close for a in A})
h=pd.DataFrame({a:D[a].reindex(I).high for a in A}); l=pd.DataFrame({a:D[a].reindex(I).low for a in A})
r=c.pct_change()
# Persistent close-location pressure: signed intraday close location, averaged over 5 sessions, lagged.
loc=((c-o)/(h-l).replace(0,np.nan)).clip(-1,1)
f=loc.rolling(5,min_periods=3).mean().shift(1)
f=f.sub(f.median(axis=1),axis=0)
print('candidate close_location_pressure dates',len(I),'assets',len(A),'coverage',round(float(f.notna().mean().mean()),4))
def run(hz):
 fr=c.pct_change(hz).shift(-hz); vals=[]; ns=[]; ds=[]
 for dt in I:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):vals.append(q);ns.append(len(z));ds.append(dt)
 v=np.array(vals); print('horizon',hz,'dates',len(v),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(v.mean(),v.mean()/v.std(ddof=1),np.mean(v>0)))
 if hz==1:
  for y in range(2020,2029):
   z=v[[x.year==y for x in ds]]
   if len(z):print('regime',y,len(z),'IC %.6f'%z.mean())
for hz in [1,5,10,20]:run(hz)
print('rank_turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean().mean()),5))
# proxy correlations, mean of datewise absolute not used as exact evidence
lib={'risk_trend':(c.pct_change(20)/r.rolling(20).std()).shift(1),'reversal':(-c.pct_change(5)/r.rolling(5).std()).shift(1),'range':((c-c.rolling(20).min())/(c.rolling(20).max()-c.rolling(20).min())).shift(1)}
for k,x in lib.items():
 q=[]
 for dt in I:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 print('proxy_abs_mean',k,round(float(np.mean(q)),4))
