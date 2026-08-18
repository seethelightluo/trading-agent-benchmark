import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 path='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(path):
  d=pd.read_csv(path)
  datecol='date' if 'date' in d else d.columns[0]
  closecol='close' if 'close' in d else 'Close'
  raw[s]=pd.Series(d[closecol].astype(float).values,index=pd.to_datetime(d[datecol]))
p=pd.DataFrame(raw).sort_index().loc[:'2033-04-29'].ffill()
r20=p.pct_change(20); vol20=p.pct_change().rolling(20).std()*np.sqrt(252); r60=p.pct_change(60)
f=r20/(vol20+1e-12); f=f.where((np.sign(r20)==np.sign(r60)) & (r60>0),0.0)
f=f.rank(axis=1,pct=True).sub(.5,axis=0)
for h in [1,3,5,10]:
 fr=p.shift(-h).div(p)-1; ics=[]; n=[]
 for dt in f.index.intersection(fr.index):
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); n.append(len(a))
 x=pd.Series(ics).replace([np.inf,-np.inf],np.nan).dropna()
 print('H',h,'dates',len(x),'avgN',round(np.mean(n),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
print('rows',len(p),'assets',len(p.columns),'coverage',round(f.notna().sum(axis=1).mean()/len(p.columns),4),'turnover',round(f.diff().abs().mean().mean(),4),'period',p.index.min(),p.index.max())
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20330429_agreement_risk_momentum_signal.csv')
