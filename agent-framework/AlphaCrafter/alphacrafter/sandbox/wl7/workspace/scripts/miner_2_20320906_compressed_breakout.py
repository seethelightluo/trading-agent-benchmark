import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; P=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None})
r=P.pct_change(); s20=r.rolling(20).std(); s60=r.rolling(60).std(); mom=r.rolling(20).sum();
# breakout direction after compressed volatility; cross-sectional interpretable
compression=(s20/(s60+1e-12)).clip(.25,2); sig=(mom/(s60+1e-12)*(1.5-compression).clip(.25,1.5)).shift(1)
print('universe',P.shape[1],'dates',len(P))
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; a=[]; n=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(x)>=8:a.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));n.append(len(x))
 z=pd.Series(a);print('H',h,'dates',len(z),'avgN',round(np.mean(n),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('coverage',round(sig.notna().sum().sum()/sig.size,4)); rk=sig.rank(axis=1,pct=True);print('turnover10',round(np.nanmean([(rk.iloc[i]-rk.iloc[i-10]).abs().mean() for i in range(10,len(rk))]),4))
f=P.shift(-10)/P-1;o=[]
for dt in sig.index:
 x=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(x)>=8:o.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
q=len(o)//3;print('thirds',[round(np.mean(o[j*q:(j+1)*q]),6) for j in range(3)])
sig.stack().rename('signal').to_csv('scripts/miner_2_20320906_compressed_breakout_signal.csv',header=True)
