import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 n=os.path.basename(p)[:-4];d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date');C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change();
# Liquidity/volatility-aware one-day reversal: fade yesterday's relative return, scaled down by recent volatility.
rel=r.sub(r.median(axis=1),axis=0); vol=r.rolling(20,min_periods=10).std(); fac=-rel/(vol+1e-8)
fac.to_csv('scripts/miner_2_20270325_volscaled1d_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h);v=[];ds=[];ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:v.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ds.append(dt);ns.append(len(x))
 return pd.Series(v,index=ds),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=a)&(s.index<=b)];print('regime',a,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
