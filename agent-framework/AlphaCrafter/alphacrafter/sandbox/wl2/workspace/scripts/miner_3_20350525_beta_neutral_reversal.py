import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-05-25')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
beta=r.rolling(120,min_periods=80).cov(m).div(m.rolling(120,min_periods=80).var(),axis=0)
resid=px.pct_change(60)-beta.mul(m.rolling(60).sum(),axis=0); f=(-resid/r.rolling(40).std()).shift(1); fr=px.shift(-40)/px-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
print('candidate beta_neutral_residual_reversal assets',len(U),'dates',len(a),'avgN',round(d.n.mean(),2),'coverage',round(f.notna().mean().mean(),4))
for label,x in [('full',a),('2026_2028',a.loc['2026':'2028']),('2029_2035',a.loc['2029':])]: print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
f.to_csv('../persistent/miner_3_20350525_beta_neutral_residual_reversal_signal.csv'); d.to_csv('../persistent/miner_3_20350525_beta_neutral_residual_reversal_ic.csv')
