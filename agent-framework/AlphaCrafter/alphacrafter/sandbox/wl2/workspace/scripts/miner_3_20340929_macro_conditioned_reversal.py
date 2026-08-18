import pandas as pd,numpy as np,os
from scipy.stats import rankdata
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):p[a]=pd.read_csv(f,usecols=['date','close'],parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index().loc['2020':];v=pd.read_csv('../persistent/index_data/VIX.csv',usecols=['date','close'],parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
base=-P.pct_change(5).shift(1); gate=(0.5+(v.shift(1)<v.shift(6)).astype(float)).to_numpy()[:,None];sig=(base.to_numpy()*gate); prices=P.values
for h in [1,5,10,20]:
 ret=pd.DataFrame(prices).pct_change(h).shift(-h).values;out=[];ns=[]; di=[]
 for i in range(len(P)):
  ok=np.isfinite(sig[i])&np.isfinite(ret[i]);n=ok.sum()
  if n>=8:
   x=rankdata(sig[i,ok]);y=rankdata(ret[i,ok]);out.append(np.corrcoef(x,y)[0,1]);ns.append(n);di.append(P.index[i])
 z=pd.Series(out,index=di);print('h',h,'dates',len(z),'n_avg',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
 if h==1:
  for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
   q=z.loc[a:b];print(a,b,len(q),q.mean(),q.mean()/q.std())
print('coverage',np.isfinite(sig).mean(),'turnover',pd.DataFrame(sig,index=P.index).rank(axis=1,pct=True).diff().abs().mean().mean())
out=pd.DataFrame(sig,index=P.index,columns=P.columns).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20340929_macro_conditioned_reversal_signal.csv',index=False)
