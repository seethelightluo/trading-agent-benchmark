import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for a in assets}
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); med=r.median(axis=1); ex=r.sub(med,axis=0)
# residual short-term reversal: negative recent residual return, damped by recent residual volatility
F=[]
for i,dt in enumerate(px.index):
 if i<65: continue
 x=ex.iloc[i-5:i].sum(); v=ex.iloc[i-20:i].std().replace(0,np.nan)
 F.append((-x/(v+1e-6)).rename(dt))
F=pd.DataFrame(F)
for h in [1,5,10,20]:
 ic=[]; ns=[]
 for dt in F.index:
  i=px.index.get_loc(dt)
  if i+h>=len(px): continue
  z=pd.concat([F.loc[dt],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(ic); print('H',h,'dates',len(x),'mean_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
rank=F.rank(axis=1,pct=True); print('turnover10',((rank-rank.shift(10)).abs().mean(axis=1)).dropna().mean(),'coverage',F.notna().mean().mean(),'dates',len(F))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
 q=[]
 for dt in F.loc[lo:hi].index:
  i=px.index.get_loc(dt)
  if i+10>=len(px):continue
  z=pd.concat([F.loc[dt],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
q=[]
for dt in F.index[-120:]:
 i=px.index.get_loc(dt)
 if i+10>=len(px):continue
 z=pd.concat([F.loc[dt],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
q=np.array(q);print('RECENT120','dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
F.to_csv('/tmp/resid_reversal.csv')
