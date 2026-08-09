import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); f=-r.rolling(30,min_periods=20).std(); y=p.pct_change(1).shift(-1); q=[];ns=[];ds=[]
for i in range(30,len(p)-1):
 z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
 if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(p.index[i])
q=np.array(q); print('dates',len(q),'range',ds[0],ds[-1],'avg',np.mean(ns),'coverage',np.sum(ns)/(len(q)*15),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1),'hit',np.mean(q>0))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q[[a<=d.strftime('%Y')<=b for d in ds]]; print('regime',a,b,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
rr=f.rank(axis=1,pct=True); print('turnover',(rr.diff().abs().mean(axis=1)/2).mean())
