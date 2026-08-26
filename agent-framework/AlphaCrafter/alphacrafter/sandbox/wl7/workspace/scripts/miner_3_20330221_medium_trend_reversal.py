import pandas as pd, numpy as np, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-02-21')
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[:end]; r=px.pct_change()
# delayed medium trend reversal: inverse 60d return / prior 40d vol
sig=-(px/px.shift(60)-1)/(r.rolling(40).std()*np.sqrt(10)+1e-12); sig=sig.shift(1)
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=np.array(vals); a=a[np.isfinite(a)]
 print('H',h,'dates',len(a),'assets_avg',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
f=px.shift(-10)/px-1; out=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8: out.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
a=pd.DataFrame(out,columns=['d','ic']); a['third']=pd.qcut(np.arange(len(a)),3,labels=False)
print('thirds',a.groupby('third').ic.mean().round(6).to_dict(),'start',a.d.min(),'end',a.d.max())
print('coverage',round(len(sig.dropna(how='all').index)/len(px.index),4),'turnover_rank_proxy',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4))
sig.to_csv('scripts/miner_3_20330221_medium_trend_reversal_signal.csv')
