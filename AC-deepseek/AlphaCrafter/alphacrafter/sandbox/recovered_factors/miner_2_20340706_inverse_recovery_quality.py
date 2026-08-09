import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for a in A}
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
P=pd.DataFrame({a:D[a].reindex(idx) for a in A}).loc[:'2034-06-21']; R=P.pct_change(fill_method=None)
# Inverse recovery-quality: favor recent losers with unstable downside and large drawdown,
# but normalize the 20d return by downside volatility to make the reversal risk-aware.
r20=P/P.shift(20)-1; down=R.where(R<0).rolling(20,min_periods=12).std(); vol=R.rolling(20,min_periods=15).std()
dd=P/P.rolling(60,min_periods=40).max()-1
F=(-(r20/(down+1e-6)+0.5*dd)).shift(1)
print('idea inverse_recovery_quality cutoff',P.index[-1].date(),'dates',len(P),'assets',len(A),'coverage',f'{F.notna().mean().mean():.6f}')
for h in [1,5,10,20]:
 xs=[];ns=[]
 for j in range(len(P)-h):
  z=pd.concat([F.iloc[j].rename('f'),(P.iloc[j+h]/P.iloc[j]-1).rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: xs.append(spearmanr(z.f,z.r).statistic); ns.append(len(z))
 x=np.array(xs); print('h',h,'dates',len(x),'meanN',f'{np.mean(ns):.2f}','IC',f'{np.mean(x):.6f}','ICIR',f'{np.mean(x)/np.std(x,ddof=1):.6f}','hit',f'{np.mean(x>0):.4f}')
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-06-21')]:
 xs=[]
 for j in range(len(P)-10):
  if not(str(P.index[j].date())>=lo and str(P.index[j].date())<=hi):continue
  z=pd.concat([F.iloc[j],(P.iloc[j+10]/P.iloc[j]-1)],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: xs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(xs); print('regime10',lo,hi,'dates',len(x),'IC',f'{np.mean(x):.6f}','ICIR',f'{np.mean(x)/np.std(x,ddof=1):.6f}')
print('turnover',f'{F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f}')
print('library_correlation NOT COMPUTED unless gates pass')
