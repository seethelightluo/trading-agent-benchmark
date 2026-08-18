import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-07-06')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=vol.std(axis=1); dz=(disp-disp.rolling(120,min_periods=60).mean())/disp.rolling(120,min_periods=60).std()
# Binary high-dispersion conditioned 5d reversal, lagged one day; zero outside stress dispersion.
f=(-px.pct_change(5)/vol).where(dz>1.0).shift(1).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20,40]:
 fr=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 d=pd.DataFrame(vals,columns=['date','ic','n']); x=d.ic
 print('h',h,'dates',len(x),'avgN',round(d.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for lab,mask in [('2025-2029',(d.date>='2025')&(d.date<='2029')),('2030-2035',d.date>='2030')]:
  y=x[mask]
  if len(y)>1: print(lab,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4),'valid_dates',len(f))
f.to_csv('../persistent/miner_2_20350706_binary_dispersion_signal.csv')
