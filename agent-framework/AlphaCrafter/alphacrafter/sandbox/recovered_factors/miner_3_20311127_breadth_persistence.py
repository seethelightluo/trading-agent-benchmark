import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 D[a]=d
px=pd.DataFrame(D).sort_index(); ret=px.pct_change()
# breadth persistence: fraction of trailing 60 sessions beating cross-sectional median return,
# with exponentially more weight on recent observations; no future data
med=ret.median(axis=1)
out=[]
for dt in px.index:
 i=px.index.get_loc(dt)
 if i<65: continue
 rr=ret.iloc[max(0,i-60):i] # through previous completed day
 ex=rr.sub(med.iloc[max(0,i-60):i],axis=0)
 w=np.exp(np.linspace(-2,0,len(ex))); w/=w.sum()
 f=(ex.gt(0).astype(float).mul(w,axis=0)).sum()
 # forward returns
 row={'date':dt}
 for a in assets: row[a]=f[a]
 out.append(row)
F=pd.DataFrame(out).set_index('date')
# only dates where 15 prices and forward available
for h in [1,5,10,20]:
 ic=[]; ns=[]
 for dt in F.index:
  i=px.index.get_loc(dt); fut=px.iloc[i+h]/px.iloc[i]-1
  z=pd.concat([F.loc[dt],fut],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(ic); print('H',h,'dates',len(x),'mean_n',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
# turnover 10d ranks
r=F.rank(axis=1,pct=True); turn=(r-r.shift(10)).abs().mean(axis=1).dropna().mean(); print('turnover10',turn,'coverage',F.notna().mean().mean(),'dates',len(F))
# regimes
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
 q=[]
 for dt in F.loc[lo:hi].index:
  i=px.index.get_loc(dt); fut=px.iloc[i+10]/px.iloc[i]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('REG',lo,hi,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
# recent
q=[]
for dt in F.index[-120:]:
 i=px.index.get_loc(dt); fut=px.iloc[i+10]/px.iloc[i]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
q=np.array(q); print('RECENT120',len(q),q.mean(),q.mean()/q.std(ddof=1))
F.to_csv('/tmp/breadth_signal.csv')
