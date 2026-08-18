import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-07-20')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=15).std(); f=(px.pct_change(20)/v).shift(1); rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
print('volscaled20 momentum dates',len(a),'avgN',round(d.n.mean(),2),'coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
for label,x in [('full',a),('2020_2025',a.loc['2020':'2025']),('2026_2030',a.loc['2026':'2030']),('2031_2035',a.loc['2031':])]:
 if len(x): print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [5,10,20,40]:
 z=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],(px.shift(-h)/px-1).loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(z);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
f.to_csv('../persistent/miner_2_20350720_volscaled20_signal.csv')
