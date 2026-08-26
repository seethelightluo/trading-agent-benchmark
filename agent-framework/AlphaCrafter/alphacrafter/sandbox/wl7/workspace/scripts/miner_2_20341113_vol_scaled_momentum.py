import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-11-13'); frames={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cut]
 frames[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(frames).sort_index(); r=p.pct_change()
f=(r.rolling(60,min_periods=40).sum()/r.rolling(20,min_periods=15).std()).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): ics.append(q); ns.append(len(z)); dates.append(dt)
 a=pd.Series(ics,index=pd.to_datetime(dates)).sort_index(); recent=a.tail(500)
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),3),'recent500',round(recent.mean(),6),'recentIR',round(recent.mean()/recent.std(ddof=1),6))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20341113_vol_scaled_momentum_signal.csv')
