import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-03-02')
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);p[a]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index().ffill();r=np.log(p).diff()
# multi-scale reversal: short shock reversal plus medium reversal, volatility normalized, lagged
s5=r.rolling(5).sum()/(r.rolling(20).std()*np.sqrt(5)); s20=r.rolling(20).sum()/(r.rolling(60).std()*np.sqrt(20))
sig=(-(0.65*s5+0.35*s20)).clip(-4,4).shift(1)
for h in [10,20,40]:
 f=np.log(p.shift(-h)/p); q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q);print(h,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean().mean())
# save 20 horizon artifacts
f=np.log(p.shift(-20)/p);out=[]
for dt in sig.index:
 for a in A:
  if pd.notna(sig.loc[dt,a]) and pd.notna(f.loc[dt,a]):out.append([dt,a,sig.loc[dt,a],f.loc[dt,a]])
pd.DataFrame(out,columns=['date','asset','signal','fwd20']).to_csv('scripts/miner_1_20340303_multiscale_reversal_20d_signal.csv',index=False)
