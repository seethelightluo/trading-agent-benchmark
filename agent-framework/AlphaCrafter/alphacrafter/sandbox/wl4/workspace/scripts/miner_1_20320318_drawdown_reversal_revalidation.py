import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv')
 d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# price-only drawdown-weighted residual reversal, with 5d shock and 60d drawdown
rows=[]
for i in range(65,len(p)-10):
 date=p.index[i]; x=r.iloc[i-5:i+1].sum(); vol=r.iloc[i-39:i+1].std(); dd=p.iloc[i]/p.iloc[i-60:i+1].max()-1
 # cross-sectional residual, contrarian, stronger for deeper drawdowns, volatility scaled
 med=x.median(); f=-(x-med)/(vol.replace(0,np.nan))*(1+(-dd).clip(0,0.5)*2)
 y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([f,y],axis=1).dropna();
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((date,ic,len(z),f))
ics=np.array([x[1] for x in rows]); print('dates',len(rows),'avgN',np.mean([x[2] for x in rows]),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics)),'hit',np.mean(ics>0),'coverage',np.mean([x[2] for x in rows])/15)
for n in [250,500,750]:
 q=ics[-n:]; print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
# rank turnover at 10 sessions
ranks=[]
for date,ic,n,f in rows: ranks.append(f.rank(pct=True))
t=[]
for j in range(10,len(ranks),10):
 a,b=ranks[j-10],ranks[j]; z=pd.concat([a,b],axis=1).dropna(); t.append(np.mean(np.abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover_proxy',np.mean(t),'decay not computed; horizon 10')
