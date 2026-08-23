import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2027-03-03');q={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date).dt.normalize();q[s]=d[d.date<=cut].drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(q,axis=1).sort_index();r=p.pct_change();v=r.rolling(20,min_periods=15).std();rng=(p.rolling(20,min_periods=15).max()-p.rolling(20,min_periods=15).min())/p
f=(-r.rolling(2,min_periods=2).sum()/(v*np.sqrt(2)+1e-12)/(1+rng)).shift(1)
def run(fr):
 z=[];ns=[]
 for i in range(len(p)):
  ok=f.iloc[i].notna()&fr.iloc[i].notna()
  if ok.sum()>=8:z.append(spearmanr(f.iloc[i][ok],fr.iloc[i][ok]).statistic);ns.append(ok.sum())
 z=np.array(z);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)
for h in [1,5,10,20]:
 a,b,c,d,e=run(p.pct_change(h).shift(-h));print(h,'dates',a,'avgN',round(b,2),'IC',round(c,8),'ICIR',round(d,6),'hit',round(e,4))
print('coverage',round(100*f.notna().sum().sum()/f.size,2),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_1_20270303_range_2d_reversal_signal.csv',index=False)
