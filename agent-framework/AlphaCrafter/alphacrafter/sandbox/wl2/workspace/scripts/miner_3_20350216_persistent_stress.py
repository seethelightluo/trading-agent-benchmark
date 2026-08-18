import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D=['XAU','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A if os.path.exists('../persistent/stock_data/'+a+'.csv')}
P=pd.DataFrame(p).sort_index();r=P.pct_change(); db=r[D].mean(axis=1)
beta=r.rolling(90,min_periods=60).cov(db).div(db.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r.sub(beta.mul(db,axis=0)); rv=res.rolling(60,min_periods=40).std().shift(1)
base=-res.rolling(30,min_periods=20).sum().shift(1)/(rv*np.sqrt(30)+1e-9)
dret=db.rolling(60,min_periods=40).sum().shift(1); dvol=db.rolling(60,min_periods=40).std().shift(1)*np.sqrt(60); z=dret/(dvol+1e-12)
persist=(db>0).rolling(40,min_periods=30).mean().shift(1)
sig=base.where((z>0.25)&(persist>0.55))
for h in [10,20,40]:
 y=P.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[dt][ok],y.loc[dt][ok]).statistic);ds.append(dt);ns.append(ok.sum())
 q=pd.Series(vals,index=ds)
 print('H',h,'dates',len(q),'avg_n',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f cov %.4f turn %.4f active %.4f recent34 %.8f recent31 %.8f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),((z>.25)&(persist>.55)).mean(),q.loc['2034':'2035'].mean(),q.loc['2031':'2035'].mean()))
 if h==40:
  out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20350216_persistent_stress_residual_signal.csv',index=False)
 print('regimes',q.loc['2026':'2030'].mean(),q.loc['2031':'2033'].mean(),q.loc['2034':'2035'].mean())
