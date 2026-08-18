import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); r=P.pct_change()
# Lagged volatility-compression breakout: short trend divided by risk, amplified when recent risk is below its medium-term baseline.
v10=r.rolling(10,min_periods=8).std().shift(1); v60=r.rolling(60,min_periods=40).std().shift(1)
r5=P.pct_change(5).shift(1)
sig=(r5/(v10*np.sqrt(5)+1e-9))*np.clip(v60/(v10+1e-9),0.5,2.0)
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h);vals=[];ns=[];ds=[]
 for d in sig.index:
  x,y=sig.loc[d],f.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=pd.Series(vals,index=ds)
 print('h',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 for label,sub in [('mid',z.loc['2024':'2029']),('recent',z.loc['2029':])]: print(label,len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20341208_compression_breakout_signal.csv',index=False)
