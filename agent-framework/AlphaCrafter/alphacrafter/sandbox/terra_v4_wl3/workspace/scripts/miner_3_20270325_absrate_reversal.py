import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); base=r.rolling(3).sum(); rel=base.sub(base.median(axis=1),axis=0)
# Absolute yield-point shock: use level changes, not percentage returns, then fade recent relative winners.
y1=close['US10Y'].diff(3); y2=close['CN10Y'].diff(3); shock=(y1+y2)/2
zs=(shock-shock.rolling(60,min_periods=30).mean())/shock.rolling(60,min_periods=30).std()
fac=-rel.mul((zs.abs()>1).astype(float),axis=0); fac.to_csv('scripts/miner_3_20270325_absrate_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));ds.append(dt)
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac),'shockdays',int((zs.abs()>1).sum()))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
