import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-06-11')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index();P=P.loc[:end];R=P.pct_change(); v=R.rolling(20).std(); f=-v
for h in [1,3,5,10,20]:
 a=[]
 for i,d in enumerate(P.index):
  if i+h>=len(P):continue
  z=pd.concat([f.loc[d].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1:a.append(spearmanr(z.x,z.y).statistic)
 a=np.array(a); print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_3_20310612_lowvol_signal.csv',index_label='date')
