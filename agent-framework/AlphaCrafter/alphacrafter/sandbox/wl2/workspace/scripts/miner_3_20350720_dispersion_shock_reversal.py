import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-07-20')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); disp=r.rolling(5,min_periods=4).std().mean(axis=1); gate=(disp>disp.rolling(120,min_periods=60).quantile(.70)).shift(1)
f=(-px.pct_change(5)/r.rolling(20,min_periods=15).std()).where(gate).shift(1)
def calc(h):
 fr=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [5,10,20,40]:
 d=calc(h); a=d.ic; print('h',h,'N',len(a),'avgN',round(d.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
d=calc(40); print('assets',len(U),'dates',len(d),'coverage',round(f.notna().mean().mean(),4),'active',round(gate.mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
for label,x in [('2020_2025',d.ic.loc['2020':'2025']),('2026_2028',d.ic.loc['2026':'2028']),('2029_2032',d.ic.loc['2029':'2032']),('2033_2035',d.ic.loc['2033':])]:
 if len(x): print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
f.to_csv('../persistent/miner_3_20350720_dispersion_shock_reversal_signal.csv'); d.to_csv('../persistent/miner_3_20350720_dispersion_shock_reversal_ic.csv')
