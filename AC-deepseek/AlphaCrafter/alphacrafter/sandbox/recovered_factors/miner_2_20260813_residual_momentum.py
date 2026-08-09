"""Validate one factor: 20d momentum of asset returns relative to same-day cross-sectional mean."""
import os,glob,json,numpy as np,pandas as pd
DATA='../persistent/stock_data'; END=pd.Timestamp('2026-08-12')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={s:pd.read_csv(f'{DATA}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float).loc[:END] for s in A}
r=pd.DataFrame({s:cs[s].pct_change() for s in A}); m=r.mean(axis=1); e=r.sub(m,axis=0)
sig=e.rolling(20,min_periods=15).sum()/e.rolling(20,min_periods=15).std().replace(0,np.nan)
def calc(h):
 fw=pd.DataFrame({s:cs[s].pct_change(h).shift(-h) for s in A}); z=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
 return pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
def met(x):return len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6),round((x.ic>0).mean(),4),round(x.n.mean(),2)
print('FACTOR cross-sectional_residual_momentum_20obs cutoff',END.date(),'cells',int(sig.count().sum()),'of',sig.size,'coverage',round(sig.count().sum()/sig.size,4))
H={h:calc(h) for h in [1,5,10,20]}
for h,x in H.items():print('H',h,'dates IC ICIR hit meanN',met(x))
for nm,lo,hi in [('2020','2020','2021'),('2021_22','2021','2023'),('2023_24','2023','2025'),('2025_now','2025','2027')]:print('REGIME',nm,met(H[10].loc[lo:hi]))
print('rank_turnover_10',round((sig.rank(pct=True)-sig.rank(pct=True).shift(10)).abs().stack().mean(),6))
def lib(p):
 fid=json.load(open(p))['factor_id']
 if 'relative_volume' in fid:
  v={s:pd.read_csv(f'{DATA}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').volume.astype(float).loc[:END] for s in A};return pd.DataFrame({s:np.log(v[s]/v[s].rolling(20,min_periods=15).mean()) for s in A})
 if 'realized_volatility' in fid:return r.rolling(20,min_periods=15).std()
 if 'volnorm_reversal' in fid:return pd.DataFrame({s:-(cs[s]/cs[s].shift(5)-1)/r[s].rolling(5,min_periods=4).std() for s in A})
 return pd.DataFrame({s:(cs[s]/cs[s].shift(20)-1)/r[s].rolling(20,min_periods=15).std() for s in A})
mx=0
for p in glob.glob('factors/*.json'):
 q=pd.concat([sig.stack().rename('x'),lib(p).stack().rename('y')],axis=1).dropna(); c=abs(q.x.corr(q.y,method='spearman'));mx=max(mx,c);print('LIBCORR',json.load(open(p))['factor_id'],round(c,6),'cells',len(q))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6))
