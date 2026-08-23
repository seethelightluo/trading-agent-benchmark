import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-02-05')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
R=P.pct_change(); mkt=R.mean(axis=1); res=R.sub(mkt,axis=0)
raw=-res.rolling(5,min_periods=4).sum()
disp=R.rolling(20,min_periods=15).std().mean(axis=1)
scale=(disp/disp.rolling(120,min_periods=40).median()).clip(0.5,2.0)
f=raw.mul(scale,axis=0).shift(1)
f.to_csv('scripts/miner_2_20310206_dispersion_scaled_residual_reversal_signal.csv',index_label='date')
for h in [5,10,20]:
 vals=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 ic=a.ic.mean(); ir=ic/a.ic.std(ddof=1)
 print('H',h,'dates',len(a),'meanN',round(a.n.mean(),2),'IC',round(ic,8),'ICIR',round(ir,8),'hit',round((a.ic>0).mean(),6),'coverage',round(a.n.mean()/15,6),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[a.index].mean(),6))
 if h==20:
  for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2031-02-05')]:
   q=a.loc[lo:hi].ic
   if len(q): print('REG',lo,round(q.mean(),8),len(q))
