import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=p.pct_change(); ret=r.rolling(60,min_periods=45).sum(); vol=r.rolling(20,min_periods=15).std()
# medium-term trend normalized by recent risk; all inputs are completed bars at signal date
f=ret/vol.replace(0,np.nan)
for h in [1,5,10]:
  y=p.pct_change(h).shift(-h); q=[]; ns=[]; dates=[]
  for i in range(60,len(p)-h):
    z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
    if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(p.index[i])
  q=np.asarray(q); print('h',h,'dates',len(q),'avg_names',round(np.mean(ns),2),'coverage',round(np.sum(ns)/(len(q)*15),4),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6),'hit',round(np.mean(q>0),4))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q[np.array([(d.strftime('%Y')>=a and d.strftime('%Y')<=b) for d in dates])]; print('regime',a,b,'n',len(z),'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6))
# rank turnover (daily signal ranks)
rr=f.rank(axis=1,pct=True); turn=(rr.diff().abs().mean(axis=1)/2).dropna(); print('turnover',round(turn.loc[rr.index.intersection(pd.DatetimeIndex(dates))].mean(),6))
