import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for a in A}
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
P=pd.DataFrame({a:D[a].reindex(idx) for a in A}); P=P.loc[:'2034-06-07']; R=P.pct_change(fill_method=None)
# Conditional short reversal: reverse the last 5d return, amplified only when broad cross-asset
# 5d dispersion is elevated relative to its trailing 60d distribution. All components are lagged.
r5=P/P.shift(5)-1; disp=R.rolling(5,min_periods=4).std().mean(axis=1); q=disp.rolling(60,min_periods=40).rank(pct=True).shift(1)
vol=R.rolling(20,min_periods=15).std(); F=(-(r5/(vol*np.sqrt(5)+1e-8))).multiply((q>0.65).astype(float),axis=0).shift(1)
print('idea conditional_high_dispersion_5d_reversal cutoff',P.index[-1].date(),'dates',len(P),'assets',len(A),'coverage',f'{F.notna().mean().mean():.6f}')
for h in [1,5,10,20]:
 xs=[]; ns=[]
 for j in range(len(P)-h):
  z=pd.concat([F.iloc[j].rename('f'),(P.iloc[j+h]/P.iloc[j]-1).rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: xs.append(spearmanr(z.f,z.r).statistic);ns.append(len(z))
 x=np.array(xs); print('h',h,'dates',len(x),'meanN',f'{np.mean(ns):.2f}','IC',f'{np.mean(x):.6f}','ICIR',f'{np.mean(x)/np.std(x,ddof=1):.6f}','hit',f'{np.mean(x>0):.4f}')
# recent regime diagnostics
h=10
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-06-07')]:
 xs=[]
 for j in range(len(P)-h):
  if not(str(P.index[j].date())>=lo and str(P.index[j].date())<=hi):continue
  z=pd.concat([F.iloc[j],(P.iloc[j+h]/P.iloc[j]-1)],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: xs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(xs); print('regime10',lo,hi,'dates',len(x),'IC',f'{np.mean(x):.6f}','ICIR',f'{np.mean(x)/np.std(x,ddof=1):.6f}')
print('turnover',f'{F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f}')
print('library_correlation NOT COMPUTED; candidate must pass IC gates before exact audit')
