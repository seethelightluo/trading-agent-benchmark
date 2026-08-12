import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for p in ['../persistent/stock_data/'+s+'.csv','../persistent/index_data/'+s+'.csv']:
  try:
   d=pd.read_csv(p,parse_dates=['date']); return d.set_index('date').close
  except: pass
 return pd.Series(dtype=float)
P=pd.DataFrame({s:load(s) for s in U}); P=P.loc[:'2026-07-15']; R=P.pct_change(); D=load('DXY').loc[:'2026-07-15'].reindex(R.index).ffill(); dr=D.pct_change()
for w in [40,60,120]:
 beta=R.rolling(w).cov(dr).div(dr.rolling(w).var(),axis=0); fac=-beta; fwd=R.shift(-1); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
   if pd.notna(q): vals.append(q);ns.append(len(z));dates.append(dt)
 a=np.array(vals); ranks=fac.rank(axis=1,pct=True)
 print('DXY inverse beta',w,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),4),'coverage',round(fac.notna().sum().sum()/(len(fac)*15),4),'turn',round(ranks.diff().abs().mean(axis=1).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
  q=[x for x,d in zip(a,dates) if lo<=str(d)[:4]<=hi]; print(' regime',lo,hi,'N',len(q),'ICIR',round(np.mean(q)/np.std(q,ddof=1),4),'IC',round(np.mean(q),5))
