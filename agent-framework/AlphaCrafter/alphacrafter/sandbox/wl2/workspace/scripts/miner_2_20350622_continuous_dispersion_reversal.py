import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-06-22')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=vol.std(axis=1); dz=(disp-disp.rolling(120,min_periods=60).mean())/disp.rolling(120,min_periods=60).std()
# Continuous, lagged dispersion-weighted short-term reversal; positive means expected higher forward return.
base=-px.pct_change(5)/vol; f=(base.mul((1+dz.clip(-1.5,1.5)),axis=0)).shift(1); f=f.replace([np.inf,-np.inf],np.nan); out=[]
for h in [5,10,20,40]:
 fr=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 d=pd.DataFrame(vals,columns=['date','ic','n']); x=d.ic
 print('h',h,'dates',len(x),'avgN',round(d.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for lab, y in [('2025-2029',x[(d.date>='2025')&(d.date<='2029')]),('2030-2035',x[d.date>='2030'])]:
  if len(y): print(lab,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
f.to_csv('../persistent/miner_2_20350622_continuous_dispersion_reversal_signal.csv')
