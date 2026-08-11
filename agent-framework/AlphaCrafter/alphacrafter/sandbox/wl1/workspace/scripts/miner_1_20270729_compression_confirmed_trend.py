import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-07-29'
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut] for s in U}
px=pd.DataFrame(P).ffill(); r=np.log(px).diff(); mom=px.shift(1).pct_change(20); v20=r.rolling(20).std()*np.sqrt(252); v60=r.rolling(60).std()*np.sqrt(252); comp=(v60-v20)/(v60+1e-12); f=mom/(v20+1e-12)*(1+.8*np.tanh(comp))
f.to_csv('scripts/miner_1_20270729_compression_confirmed_trend_signal.csv')
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(vals).dropna();print(h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'cutoff',px.index.max())
