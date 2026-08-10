import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4];d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date');C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index();r=close.pct_change()
# Volatility-term-structure factor: assets whose recent 5d realized risk is low
# relative to their 30d baseline receive higher scores (risk normalization).
v5=r.rolling(5,min_periods=5).std();v30=r.rolling(30,min_periods=15).std()
raw=-np.log((v5+1e-10)/(v30+1e-10)).replace([np.inf,-np.inf],np.nan)
fac=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1).replace(0,np.nan),axis=0)
fac.to_csv('scripts/miner_2_20270325_volterm_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h);a=[];n=[];ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:a.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);n.append(len(x));ds.append(dt)
 s=pd.Series(a,index=ds);return s,n
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
s,_=ev(1)
for q in np.array_split(s,4):print('block',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
