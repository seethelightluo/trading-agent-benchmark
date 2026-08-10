import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-02-25')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x['date']); D[s]=x[x.date<=cutoff].sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); f=(P.pct_change(40)/r.rolling(20).std()).shift(1)
for h in [1,5,10]:
 rows=[]; ns=[]; y=P.pct_change(h).shift(-h)
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8: rows.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a))
 q=np.asarray(rows); ic=np.nanmean(q); ir=ic/np.nanstd(q,ddof=1)*np.sqrt(len(q)); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(ic,ir,np.mean(q>0)))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
y=P.pct_change().shift(-1)
for yr in sorted(set(f.index.year)):
 z=[]
 for dt in f.index[f.index.year==yr]:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 if len(z)>1: print('year',yr,'dates',len(z),'IC %.5f ICIR %.4f'%(np.mean(z),np.mean(z)/np.std(z,ddof=1)*np.sqrt(len(z))))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_volnorm_mom40.csv',index=False)
