import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index().loc[:'2033-10-14']; r=prices.pct_change()
# interpretable risk-adjusted medium momentum, lagged one day
sig=(prices.pct_change(20)/r.rolling(20).std()).shift(1)
# neutralize cross-sectional level each date
sig=sig.sub(sig.mean(axis=1),axis=0)
for h in [1,3,5,10,20]:
 f=prices.pct_change(h).shift(-h)
 vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=f.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
print('coverage',sig.notna().sum(axis=1).ge(8).mean(),'assets',len(prices.columns),'span',prices.index.min(),prices.index.max())
# artifact used for admission horizon 10
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20331014_risk_adjusted_momentum_signal.csv',index_label='date')
