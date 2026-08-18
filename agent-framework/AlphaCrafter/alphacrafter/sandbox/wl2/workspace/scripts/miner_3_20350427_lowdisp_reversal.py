import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-04-27')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date'); D[s]=x.close.loc[:CUT]
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); trend=px.pct_change(60); vol=r.rolling(40).std(); disp=r.rolling(20).std().mean(axis=1); active=(disp<disp.rolling(120).median()).astype(float); f=(-trend/vol).mul(active,axis=0).shift(1)
for h in [5,10,20,40]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print(h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
fr=px.shift(-40)/px-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
rr=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for a,b in [('2020','2022-12-31'),('2023','2028-12-31'),('2029','2035-04-27')]:
 q=rr.loc[a:b].ic; print('regime',a,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('assets',len(U),'dates',len(rr),'coverage',round(f.notna().mean().mean(),4),'activation',round(active.loc[:CUT].mean(),4))
f.to_csv('../persistent/miner_3_20350427_lowdisp_reversal_signal.csv'); rr.to_csv('../persistent/miner_3_20350427_lowdisp_reversal_ic.csv')
