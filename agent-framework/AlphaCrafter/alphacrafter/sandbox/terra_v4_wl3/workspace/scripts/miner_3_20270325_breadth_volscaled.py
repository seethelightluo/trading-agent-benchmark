import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); ret=close.pct_change(); r20=close.pct_change(20); r5=close.pct_change(5)
breadth=(r20>0).mean(axis=1); bull=breadth>=0.5
v20=ret.rolling(20).std(); v5=ret.rolling(5).std()
def z(x): return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
fac=z(r20/v20).where(bull, -z(r5/v5)); fac.to_csv('scripts/miner_3_20270325_breadth_volscaled_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));dates.append(dt)
 return pd.Series(vals,index=dates),ns
print('assets',len(C),'rows',len(fac),'bull_days',int(bull.sum()))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
