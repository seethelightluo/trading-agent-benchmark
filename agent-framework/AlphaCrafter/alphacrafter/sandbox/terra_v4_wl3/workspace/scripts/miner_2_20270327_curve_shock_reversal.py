import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change();
# Yield-curve shock reversal: fade relative 3d weakness when US-CN 10y spread changes sharply.
curve=close['US10Y']-close['CN10Y']
shock=curve.diff(3).abs()
q=shock.rolling(252,min_periods=60).quantile(.80).shift(1)
gate=(shock>q).astype(float)
r3=close.pct_change(3); rel=r3.sub(r3.median(axis=1),axis=0)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=(-z).mul(gate,axis=0); fac.to_csv('scripts/miner_2_20270327_curve_shock_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ds.append(dt); ns.append(len(x))
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac),'active',int(gate.sum()))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2) if n else 0,'IC',round(s.mean(),7) if len(s) else np.nan,'ICIR',round(s.mean()/s.std(ddof=1),7) if len(s)>1 else np.nan,'hit',round((s>0).mean(),4) if len(s) else np.nan)
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   x=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(x),round(x.mean(),7) if len(x) else np.nan)
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
