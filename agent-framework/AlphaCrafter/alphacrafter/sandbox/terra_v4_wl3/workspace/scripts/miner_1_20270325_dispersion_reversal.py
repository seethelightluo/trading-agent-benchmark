import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv')
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Dispersion-conditioned short-term reversal: fade relative 3d returns, activated
# during broad cross-asset dispersion, when reversal opportunities are larger.
ret3=r.rolling(3).sum(); rel=ret3.sub(ret3.median(axis=1),axis=0)
disp=ret3.std(axis=1)
baseline=disp.rolling(60,min_periods=30).median().replace(0,np.nan)
activation=(disp/baseline).clip(0,3)
fac=-rel.mul(activation,axis=0)
fac.to_csv('scripts/miner_1_20270325_dispersion_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); dates.append(dt); ns.append(len(x))
 return pd.Series(vals,index=dates),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),round(q.mean(),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/len(C),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
