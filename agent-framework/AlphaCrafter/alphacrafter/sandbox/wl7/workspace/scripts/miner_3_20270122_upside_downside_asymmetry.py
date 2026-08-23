import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; sig={}; fwd={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2027-01-22']
 r=d.pct_change(); dn=(-r.clip(upper=0)).rolling(20,min_periods=20).mean()
 sig[a]=(r.rolling(10,min_periods=10).sum()/(dn+1e-6)); fwd[a]=d.shift(-1)/d-1
S=pd.DataFrame(sig); Y=pd.DataFrame(fwd); rows=[]
for dt in S.index:
 z=pd.concat([S.loc[dt].rename('s'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.s,z.y).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(q.ic>0).mean())
for h in [5,10,20]:
 yy={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2027-01-22'].shift(-h)/pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2027-01-22']-1 for a in A}; YY=pd.DataFrame(yy); rr=[]
 for dt in S.index:
  z=pd.concat([S.loc[dt].rename('s'),YY.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.s,z.y).statistic)
 rr=pd.Series(rr); print('decay',h,rr.mean(),len(rr))
print('coverage_valid_dates',S.notna().sum(axis=1).mean()/15,'turnover',S.rank(pct=True).diff().abs().mean(axis=1).mean())
S.stack().rename('signal').to_csv('scripts/miner_3_20270122_upside_downside_asymmetry_signal.csv')
