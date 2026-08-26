import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-10-13')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# one simple, interpretable candidate: 20D cross-asset reversal, downside-vol normalized
ret=p/p.shift(20)-1
vol=r.rolling(40).std()*np.sqrt(252)
f=(-ret/(vol+0.05)).shift(1).clip(-5,5)
rows=[]
for h in [10,20,40,60]:
  ic=[]; ns=[]
  fr=p.shift(-h)/p-1
  for dt in f.index:
   a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
   if len(z)>=8:
    ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  x=pd.Series(ic).dropna(); print(h,len(x),np.mean(ns),x.mean(),x.std(ddof=1),x.mean()/x.std(ddof=1),np.mean(x>0))
# artifacts for admission horizon 20
fr=p.shift(-20)/p-1
out=f.stack().rename('signal').to_frame(); out['forward_return_20d']=fr.stack(); out=out.reset_index().rename(columns={'level_0':'date','level_1':'symbol'}); out=out.dropna(); out.to_csv('scripts/miner_1_20331013_short_reversal_signal.csv',index=False)
print('artifact',len(out),'coverage',f.notna().sum().sum()/f.size,'turnover',f.diff().abs().mean().mean())
