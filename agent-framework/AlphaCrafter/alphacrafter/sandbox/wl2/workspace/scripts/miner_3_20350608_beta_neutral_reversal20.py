import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-06-08')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
beta=r.rolling(120,min_periods=80).cov(m).div(m.rolling(120,min_periods=80).var(),axis=0)
# Shorter residual reversal: remove 120d benchmark beta exposure from 20d return, risk scale, lag one day.
resid=px.pct_change(20)-beta.mul(m.rolling(20).sum(),axis=0); f=(-resid/r.rolling(40).std()).shift(1); fr=px.shift(-20)/px-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
print('candidate beta_neutral_residual_reversal_20d assets',len(U),'dates',len(a),'avgN',round(d.n.mean(),2),'coverage',round(f.notna().mean().mean(),4),'turnover_proxy',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
for label,x in [('full',a),('2020_2025',a.loc['2020':'2025']),('2026_2028',a.loc['2026':'2028']),('2029_2035',a.loc['2029':])]:
 if len(x): print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [5,10,20,40]:
 fh=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay_h',h,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
f.to_csv('../persistent/miner_3_20350608_beta_neutral_residual_reversal20_signal.csv'); d.to_csv('../persistent/miner_3_20350608_beta_neutral_residual_reversal20_ic.csv')
