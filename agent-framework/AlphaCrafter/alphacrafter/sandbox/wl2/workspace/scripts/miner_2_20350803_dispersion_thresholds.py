import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-08-03')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index()
r=px.pct_change(); csdisp=r.std(axis=1,ddof=1); z=(csdisp-csdisp.rolling(120,min_periods=60).mean())/csdisp.rolling(120,min_periods=60).std()
# Dispersion shock: cross-sectional 5d reversal, volatility scaled, activated by lagged dispersion z-score.
rev=-px.pct_change(5); vol=r.rolling(20,min_periods=15).std(); raw=rev/vol
fr=px.shift(-40)/px-1
for th in [1.0,1.25]:
 f=raw.where(z.shift(1)>th).shift(1); rows=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
 d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
 print('threshold',th,'dates',len(d),'avgN',round(d.n.mean(),2),'IC40',round(a.mean(),6),'ICIR40',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'active',round((z.shift(1)>th).mean(),4),'coverage',round(f.notna().mean().mean(),4))
 for lab,x in [('2020_2025',a.loc['2020':'2025']),('2026_2028',a.loc['2026':'2028']),('2029_2032',a.loc['2029':'2032']),('2033_2035',a.loc['2033':])]:
  if len(x): print(lab,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
 if th==1.0: f.to_csv('../persistent/miner_2_20350803_dispersion100_signal.csv'); d.to_csv('../persistent/miner_2_20350803_dispersion100_ic.csv')
 if th==1.25: f.to_csv('../persistent/miner_2_20350803_dispersion125_signal.csv'); d.to_csv('../persistent/miner_2_20350803_dispersion125_ic.csv')
# turnover of ranks for 1.0
f=raw.where(z.shift(1)>1.0).shift(1)
print('rank_turnover_proxy',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
