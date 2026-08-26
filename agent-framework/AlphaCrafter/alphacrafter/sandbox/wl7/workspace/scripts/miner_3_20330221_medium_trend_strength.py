import pandas as pd, numpy as np, glob, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index(); r=px.pct_change()
# candidate: medium trend strength, 60d return normalized by 40d vol, lagged one day
sig=(px/px.shift(60)-1)/(r.rolling(40).std()*np.sqrt(10)+1e-12)
sig=sig.shift(1)
rows=[]
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1
 ics=[]; turns=[]; cov=[]
 for dt in px.index:
  z=sig.loc[dt]; y=f.loc[dt]; q=pd.concat([z,y],axis=1).dropna()
  if len(q)>=8:
   ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); cov.append(len(q)/15)
 # signal turnover rank changes daily
 for dt in px.index[1:]:
  q=pd.concat([sig.shift(1).loc[dt],sig.loc[dt]],axis=1).dropna()
  if len(q)>=8: turns.append((q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean())/len(q))
 a=np.array(ics); a=a[np.isfinite(a)]
 print('H',h,'dates',len(a),'assets_avg',round(np.mean(np.array(cov)*15),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'turn',round(np.mean(turns),4))
# thirds H10
f=px.shift(-10)/px-1; a=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8:a.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
a=pd.DataFrame(a,columns=['d','ic']); a['third']=pd.qcut(np.arange(len(a)),3,labels=False)
print('thirds',a.groupby('third').ic.mean().round(6).to_dict(),'start',a.d.min(),'end',a.d.max())
# signal artifact
sig.to_csv('scripts/miner_3_20330221_medium_trend_strength_signal.csv')
