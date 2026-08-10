import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); names=[os.path.basename(p)[:-4] for p in files]
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Volatility-adjusted medium momentum, with a short-term reversal overlay.
# The 20-session trend is scaled by trailing 20-session realized volatility;
# subtracting the 3-session return avoids buying temporarily overextended leaders.
trend=close.pct_change(20)
vol=r.rolling(20).std()*np.sqrt(252)
raw=trend/vol.replace(0,np.nan)-0.35*close.pct_change(3)
fac=raw.sub(raw.median(axis=1),axis=0)
fac.to_csv('scripts/miner_2_20270325_voladj_momentum_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); v=[];ds=[];ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   v.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ds.append(dt);ns.append(len(x))
 return pd.Series(v,index=ds),ns
print('assets',len(names),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(names),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
