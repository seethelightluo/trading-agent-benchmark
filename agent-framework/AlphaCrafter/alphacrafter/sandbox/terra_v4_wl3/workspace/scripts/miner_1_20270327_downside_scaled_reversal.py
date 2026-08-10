import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
down=r.where(r<0).rolling(20,min_periods=10).std(); up=r.where(r>0).rolling(20,min_periods=10).std()
# Downside-risk-adjusted short-term reversal: recent 5d loss/gain reversed, scaled by downside volatility.
fac=-r.rolling(5,min_periods=5).sum().div(down.replace(0,np.nan))
fac.to_csv('scripts/miner_1_20270327_downside_scaled_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); v=[]; ns=[]; ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   v.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));ds.append(dt)
 return pd.Series(v,index=ds),ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo+'-'+hi,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
