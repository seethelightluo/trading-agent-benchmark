import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[s]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r5=close.pct_change(5)
# Macro-conditioned reversal: fade five-day relative moves only when observation-only VIX is elevated versus its 60d history.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close
v=v.reindex(close.index).ffill(); vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
rel=r5.sub(r5.median(axis=1),axis=0); base=-rel
act=vz.clip(lower=0,upper=2)/2
fac=base.mul(act,axis=0)
fac.to_csv('scripts/miner_3_20270325_vix5_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); o=[];n=[];ix=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:o.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);n.append(len(x));ix.append(dt)
 return pd.Series(o,index=ix),n
print('assets',len(C),'rows',len(fac),'active',int((act>0).sum()))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
