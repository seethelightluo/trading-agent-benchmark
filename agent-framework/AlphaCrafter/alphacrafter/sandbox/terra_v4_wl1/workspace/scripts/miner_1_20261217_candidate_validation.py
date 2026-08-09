import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); O=pd.DataFrame({s:D[s].open for s in U}).reindex(P.index); H=pd.DataFrame({s:D[s].high for s in U}).reindex(P.index); L=pd.DataFrame({s:D[s].low for s in U}).reindex(P.index); V=pd.DataFrame({s:D[s].volume for s in U}).reindex(P.index); R=P.pct_change(); tr=pd.concat([(H-L).stack(),(H-P.shift()).abs().stack(),(L-P.shift()).abs().stack()],axis=1).max(axis=1).unstack()
fac={'range_efficiency_20':P.pct_change(20)/(tr.rolling(20,min_periods=15).sum()/P.shift(1)),'volume_trend_continuation':P.pct_change(20)*np.log1p((V.shift(1)/(V.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)),'vol_managed_momentum_20':P.pct_change(20)/(R.rolling(20,min_periods=15).std()*np.sqrt(20)),'return_autocorr_reversal':-R.rolling(20,min_periods=15).corr(R.shift(1))}
for name,F in fac.items():
 print('\n',name)
 for h in [1,5,10]:
  y=P.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
  for d in P.index:
   q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q)); dates.append(d)
  a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
 print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
olds={'mom20':P.pct_change(20),'rev5':-P.pct_change(5),'clv':((P-L)-(H-P))/(H-L).replace(0,np.nan)}
for n,F in fac.items():
 for on,old in olds.items():
  z=pd.concat([F.stack().rename('f'),old.stack().rename('o')],axis=1).dropna(); print('corr',n,on,round(z.f.corr(z.o),4))
print('period',P.index.min(),P.index.max(),'symbols',len(U))
