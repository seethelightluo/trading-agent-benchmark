import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2030-05-30')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change(); r5=p.pct_change(5); resid=r5.sub(r5.median(axis=1),axis=0); down=r.where(r<0,0).rolling(20).std(); up=r.where(r>0,0).rolling(20).std(); asym=(down/(up+1e-4)).clip(.5,4); vol=r.rolling(20).std()*np.sqrt(252); sig=(-resid.clip(upper=0)*asym/(vol+.01)).shift(1)
for h in [5,10,20,40]:
 y=p.shift(-h)/p-1; a=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q)
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',15,'coverage',1.0,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover_proxy',round((sig.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6)); sig.index.name='date';sig.to_csv('scripts/miner_1_20300530_downside_asymmetry_rebound_signal.csv')
