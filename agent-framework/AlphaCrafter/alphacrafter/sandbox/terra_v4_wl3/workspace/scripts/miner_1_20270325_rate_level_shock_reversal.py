import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv')
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); ret=close.pct_change()
# Absolute yield-point shock avoids percentage-return degeneracy in near-flat yield series.
yshock=(close['US10Y'].diff(3)+close['CN10Y'].diff(3))/2
scale=(yshock.abs()/yshock.abs().rolling(60).median().replace(0,np.nan)).clip(0,3)
base=ret.rolling(3).sum(); rel=base.sub(base.median(axis=1),axis=0)
fac=-rel*scale
fac.to_csv('scripts/miner_1_20270325_rate_level_shock_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ds.append(dt); ns.append(len(x))
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2) if n else 0,'IC',round(s.mean(),7) if len(s) else 'nan','ICIR',round(s.mean()/s.std(ddof=1),7) if len(s)>1 else 'nan','hit',round((s>0).mean(),4) if len(s) else 'nan')
 if h==1:
  for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',label,'n',len(q),'IC',round(q.mean(),7) if len(q) else 'nan')
print('coverage',round(fac.notna().sum(axis=1).mean()/len(C),6),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
