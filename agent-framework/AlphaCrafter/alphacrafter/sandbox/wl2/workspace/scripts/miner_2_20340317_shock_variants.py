import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); P=P.loc[P.index<='2034-03-16']; r=P.pct_change(); v=r.rolling(20,min_periods=15).std(); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); gate=(r.std(axis=1)>disp).astype(float)
Fs={'shock_reversal':-r.div(v).mul(gate,axis=0),'tail_reversal':-r*(1+(r.abs().gt(2*v)).astype(float)),'range_reversal':-P.pct_change(3).div(v*np.sqrt(3)).mul(gate,axis=0)}
y=r.shift(-1)
for name,F in Fs.items():
 rows=[]
 for dt in P.index:
  q=pd.concat([F.loc[dt],y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 D=pd.DataFrame(rows,columns=['date','ic']).set_index('date'); a=D.ic.values
 print(name,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for st in ['2026-01-01','2031-01-01','2033-01-01']:
  b=D[D.index>=st].ic; print(' ',st,len(b),b.mean(),b.mean()/b.std(ddof=1))
