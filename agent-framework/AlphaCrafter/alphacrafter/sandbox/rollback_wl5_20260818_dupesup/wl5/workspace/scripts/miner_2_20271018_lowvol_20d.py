import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-10-18')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date'); x=x[x.date<=end].set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Factor: inverse 20d realized volatility, cross-sectionally ranking stable assets
f=-r.rolling(20,min_periods=20).std()
for h in [1,5,10]:
 q=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 a=np.array(vals); print('h',h,'dates',len(a),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round(np.mean(a>0),4),'std',round(a.std(ddof=1),6))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  z=a[[lo<=d.year<=hi for d in dates]]; print(' regime',lo,hi,'n',len(z),'ic',round(z.mean(),6) if len(z) else None,'icir',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
# average rank turnover, valid coverage
print('overall period',p.index.min().date(),p.index.max().date(),'factor coverage',round(f.notna().sum().sum()/f.size,4),'rank turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
# artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20271018_lowvol_20d_signal.csv',index=False)
