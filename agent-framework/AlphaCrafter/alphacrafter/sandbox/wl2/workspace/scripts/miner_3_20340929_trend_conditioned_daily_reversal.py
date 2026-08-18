import pandas as pd,numpy as np,os
from scipy.stats import rankdata
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):p[a]=pd.read_csv(f,usecols=['date','close'],parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index().loc['2020':];r=P.pct_change();trend=r.mean(axis=1).rolling(20).sum().shift(1);sig=(-r.shift(1)*(1+0.5*(trend>0).astype(float)).to_numpy()[:,None]);prices=P.values
for h in [1,5,10]:
 ret=pd.DataFrame(prices).pct_change(h).shift(-h).values;z=[];ns=[];di=[]
 for i in range(len(P)):
  ok=np.isfinite(sig[i])&np.isfinite(ret[i]);n=ok.sum()
  if n>=8:z.append(np.corrcoef(rankdata(sig[i,ok]),rankdata(ret[i,ok]))[0,1]);ns.append(n);di.append(P.index[i])
 z=pd.Series(z,index=di);print('h',h,'dates',len(z),'n',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
print('coverage',np.isfinite(sig).mean(),'turnover',pd.DataFrame(sig,index=P.index).rank(axis=1,pct=True).diff().abs().mean().mean());pd.DataFrame(sig,index=P.index,columns=P.columns).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('../persistent/miner_3_20340929_trend_conditioned_daily_reversal_signal.csv',index=False)
