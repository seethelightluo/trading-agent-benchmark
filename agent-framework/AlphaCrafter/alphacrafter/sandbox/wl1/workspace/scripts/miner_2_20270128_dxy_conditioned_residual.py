import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-01-27'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(20); eq=R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1); resid=R.sub(eq,axis=0)
# DXY trend regime: relative strength is favored when dollar trend is weak; reverse when strong.
dollar=dxy.pct_change(20).rolling(10,min_periods=5).mean(); state=-np.tanh(dollar*8)
f=resid.mul(state,axis=0).shift(1)
print('candidate DXY-conditioned equity-residual momentum; signal lag 1')
for h in [5,10,20]:
 a=[]; ns=[]
 for dt in P.loc[:cut].index:
  fut=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x): a.append(x);ns.append(len(q))
 a=np.asarray(a); print('h',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
valid=f.loc[:cut].notna().sum(axis=1); print('coverage',valid.mean()/15,'avg_n',valid.mean(),'turnover',f.loc[:cut].rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('annual 10d')
out=[]
for dt in P.loc[:cut].index:
 q=pd.concat([f.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: out.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
d=pd.DataFrame(out,columns=['date','ic']).set_index('date'); print(d.groupby(d.index.year).ic.agg(['count','mean']).to_string())
