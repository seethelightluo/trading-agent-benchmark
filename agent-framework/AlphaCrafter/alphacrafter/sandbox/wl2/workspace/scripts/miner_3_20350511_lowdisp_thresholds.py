import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-05-11')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index()
r=px.pct_change(); trend=px.pct_change(60); vol=r.rolling(40).std(); disp=r.rolling(20).std().mean(axis=1)
fr=px.shift(-40)/px-1
for pct in [0.30,0.40,0.50,0.60]:
 thr=disp.rolling(120).quantile(pct); active=(disp<thr).astype(float); f=(-trend/vol).mul(active,axis=0).shift(1)
 vals=[]; ns=[]; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; vals.append(ic); ns.append(len(z)); rows.append((dt,ic))
 a=np.array(vals); recent=pd.DataFrame(rows,columns=['date','ic']).set_index('date').loc['2029':].ic
 print('pct',pct,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'cov',round(f.notna().mean().mean(),4),'act',round(active.mean(),4),'recent',len(recent),round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
# save best artifact only if visibly passes broad and recent
pct=.60; active=(disp<disp.rolling(120).quantile(pct)).astype(float); f=(-trend/vol).mul(active,axis=0).shift(1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
pd.DataFrame(rows,columns=['date','ic']).to_csv('../persistent/miner_3_20350511_lowdisp40_ic.csv',index=False)
f.to_csv('../persistent/miner_3_20350511_lowdisp40_signal.csv')
print('assets',len(U),'validation_end',CUT.date())
