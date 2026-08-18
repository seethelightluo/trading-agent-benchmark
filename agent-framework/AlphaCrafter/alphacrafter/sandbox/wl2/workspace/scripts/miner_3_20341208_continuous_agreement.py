import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); r=P.pct_change()
v20=r.rolling(20,min_periods=15).std().shift(1); v40=r.rolling(40,min_periods=30).std().shift(1)
r10=P.pct_change(10).shift(1); r40=P.pct_change(40).shift(1)
# continuous multi-horizon agreement: fast risk-scaled trend weighted by bounded slow trend strength
sig=(r10/(v20*np.sqrt(10)+1e-9))*np.tanh(r40/(v40*np.sqrt(40)+1e-9))
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for d in sig.index:
  x,y=sig.loc[d],f.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=pd.Series(vals,index=ds)
 print('h',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 for label,sub in [('early',z.loc[:'2024']),('mid',z.loc['2024':'2029']),('recent',z.loc['2029':])]: print(label,len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20341208_continuous_agreement_signal.csv',index=False)
