import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv')
C={os.path.basename(p)[:-4]:pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date').close for p in files}
close=pd.DataFrame(C).sort_index(); close=close.loc[:cut]; r=close.pct_change();
# Volatility-scaled medium-term relative momentum: 20d return divided by 20d realized vol,
# centered cross-sectionally to remove common market direction.
ret=close.pct_change(20); vol=r.rolling(20).std()*np.sqrt(20); raw=ret/vol.replace(0,np.nan)
fac=raw.sub(raw.median(axis=1),axis=0)
fac.to_csv('scripts/miner_2_20270325_volscaled_momentum20_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ds.append(dt); ns.append(len(x))
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
s,n=ev(1)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
