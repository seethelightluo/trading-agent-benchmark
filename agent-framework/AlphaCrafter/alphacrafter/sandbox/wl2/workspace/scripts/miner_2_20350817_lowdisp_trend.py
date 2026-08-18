import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-08-17')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); trend=px.pct_change(60); vol=r.rolling(40,min_periods=20).std(); disp=r.rolling(20,min_periods=15).std().mean(axis=1); med=disp.rolling(120,min_periods=60).median(); active=(disp<med).astype(float); f=(trend/vol).mul(active,axis=0).shift(1)
def calc(h):
 fr=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [5,10,20,40]:
 d=calc(h); a=d.ic; print('h',h,'N',len(a),'avgN',round(d.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
d=calc(40); print('assets',len(U),'dates',len(d),'coverage',round(f.notna().mean().mean(),4),'activation',round(active.mean(),4));
f.to_csv('../persistent/miner_2_20350817_lowdisp_trend_signal.csv'); d.to_csv('../persistent/miner_2_20350817_lowdisp_trend_ic.csv')
for label,lo,hi in [('2020-2025','2020','2025-12-31'),('2026-2029','2026','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2035','2033','2035-12-31')]:
 x=d.loc[pd.Timestamp(lo):pd.Timestamp(hi)].ic; print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
