import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p=Path('../persistent/stock_data')
w=pd.DataFrame({s:pd.read_csv(p/f'{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=w.pct_change();
# resilience: low downside volatility, conditioned on positive medium trend
v=r.rolling(30,min_periods=25).std(); down=r.where(r<0).rolling(30,min_periods=25).std(); trend=w.pct_change(60)
f=trend/(down*np.sqrt(30)+1e-12) - .35*v*np.sqrt(30)
for h in [5,10,20]:
 a=[]; ns=[]
 for i in range(len(w)-h):
  z=pd.concat([f.iloc[i],(w.iloc[i+h]/w.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for yr in range(2026,2033):
 a=[]
 for i in range(len(w)-10):
  if w.index[i].year!=yr:continue
  z=pd.concat([f.iloc[i],w.iloc[i+10]/w.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(yr,len(a),np.mean(a))
