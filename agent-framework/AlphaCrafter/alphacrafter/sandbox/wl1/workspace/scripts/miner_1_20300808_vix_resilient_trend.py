import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
v=get_index_daily_data('VIX',2600)
V=v.set_index('date')['close'].astype(float).reindex(P.index).ffill(); vr=V.pct_change()
trend=P.pct_change(60); vol=r.rolling(30).std()*np.sqrt(252); beta=r.rolling(60).corr(vr)
stress=(vr.rolling(60).mean()+.5*vr.rolling(60).std()); flag=(vr>stress).astype(float)
f=(trend/(vol+1e-8) - beta.mul(flag,axis=0)*.75).shift(1)
for h in [1,5,10,20]:
 fr=P.shift(-h)/P-1; z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(m,m/sd,(q.ic>0).mean()))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).mean(),6));print('period',P.index.min(),P.index.max(),'assets',len(U));f.to_csv('scripts/miner_1_20300808_vix_resilient_trend_signal.csv')
