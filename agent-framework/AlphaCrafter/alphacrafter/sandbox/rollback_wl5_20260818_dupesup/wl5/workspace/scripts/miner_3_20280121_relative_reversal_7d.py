import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-20'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:end].ffill()
r=P.pct_change(7); f=-r.sub(r.median(axis=1),axis=0); y=P.shift(-10)/P-1

def c(x,sl=slice(None)):
 a=[];n=[]
 for d in x.loc[sl].index:
  z=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 a=np.array(a);return len(a),np.mean(n),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('7d ALL',c(f))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-20')]:print('REG',lo,c(f,slice(lo,hi)))
rk=f.rank(axis=1,pct=True);print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(rk-rk.shift()).abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_3_20280121_relative_reversal_7d_signal.csv')
