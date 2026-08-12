import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:get_stock_daily_data(s,days=3600).set_index('date')['close'] for s in U}).sort_index().ffill();r=px.pct_change();
# volatility-normalized medium trend, no future data
f=(px.pct_change(20)/r.rolling(40,min_periods=30).std()).shift(1)
fr={h:px.pct_change(h).shift(-h) for h in [1,5,10,20]}
for h,y in fr.items():
 vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(vals).dropna();print(h,len(x),np.mean(ns),x.mean(),x.mean()/x.std(),(x>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20291213_volnorm_signal.csv')
