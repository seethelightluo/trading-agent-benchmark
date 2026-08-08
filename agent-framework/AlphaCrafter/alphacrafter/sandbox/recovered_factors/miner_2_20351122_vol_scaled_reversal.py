import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for a in A}).sort_index(); R=P.pct_change(fill_method=None)
# Candidate: lagged short-horizon reversal, scaled by recent volatility and damped in highly dispersed sessions.
rev=-(P/P.shift(3)-1); vol=R.rolling(20,min_periods=15).std(); disp=R.std(axis=1).rolling(20,min_periods=15).mean(); damp=(1/(1+disp/disp.rolling(120,min_periods=60).median())).shift(1)
F=(rev/(vol+1e-12)).mul(damp,axis=0).shift(1); F=F.sub(F.mean(axis=1),axis=0)
print('idea=vol-scaled 3d reversal with dispersion damping rows',len(P),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',float(F.notna().mean().mean()))
def ev(h,sub=F):
 fw=P.shift(-h)/P-1; out=[]; ns=[]; ds=[]
 for t in sub.index:
  q=pd.concat([sub.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: out.append(spearmanr(q.f,q.r).statistic);ns.append(len(q));ds.append(t)
 x=np.asarray(out); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns)),pd.Series(x,index=ds)
for h in [1,5,10,20]:
 z=ev(h);print('H',h,'dates',z[0],'IC',round(z[1],6),'ICIR',round(z[2],6),'hit',round(z[3],4),'meanN',round(z[4],2))
z=ev(5)[-1]
for lo,hi in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-11-21')]:
 x=z.loc[lo:hi];print('regime',lo,hi,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover_daily_rank',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
print('LIBRARY_AUDIT=FAILED exact admitted signal histories unavailable; no persistence')
