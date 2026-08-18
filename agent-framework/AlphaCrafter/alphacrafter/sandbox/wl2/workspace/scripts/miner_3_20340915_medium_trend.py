import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); r=P.pct_change();
# medium trend: 60d return excluding latest 5d, volatility normalized, lagged
sig=(P.shift(5)/P.shift(65)-1)/(r.rolling(60,min_periods=40).std()*np.sqrt(60)+1e-9); sig=sig.shift(1)
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for d in sig.index:
  x,y=sig.loc[d],f.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());dates.append(d)
 z=pd.Series(vals,index=dates); print('h',h,'dates',len(z),'n',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 print('recent',z.loc['2026':].mean(),z.loc['2026':].mean()/z.loc['2026':].std(ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20340915_medium_trend_signal.csv',index=False)
