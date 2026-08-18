import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(f); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); dates=p.index
# one interpretable idea: positive skew / asymmetric return shape, 20d rolling skewness
fac=r.rolling(20,min_periods=15).skew()
# test horizons
for h in [1,5,10]:
 vals=[]; ics=[]; turns=[]; cov=[]
 for i in range(20,len(p)-h):
  a=fac.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1
  z=pd.concat([a,y],axis=1).dropna();
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); vals.append(len(z)); cov.append(len(z)/15)
 print(h,'N',len(ics),'mean names',np.mean(vals),'coverage',np.mean(cov),'IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(np.array(ics)>0))
# turnover ranks / standardized signal
print('turnover',np.nanmean(np.abs(fac.rank(pct=True).diff()).stack()))
# regime annual
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 x=[]
 for i in range(20,len(p)-1):
  if not (str(p.index[i].year) >= a and str(p.index[i].year)<=b): continue
  z=pd.concat([fac.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(a,b,np.nanmean(x),len(x))
# corr existing proxies
m=r.rolling(20).sum(); print('corr mom',fac.stack().corr(m.stack()))
