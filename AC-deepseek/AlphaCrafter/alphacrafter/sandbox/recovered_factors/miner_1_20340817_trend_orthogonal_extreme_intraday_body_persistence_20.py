import numpy as np
import pandas as pd
from scipy.stats import spearmanr
# One idea: persistence of unusually directional intraday bodies, rather than average close location.
END=pd.Timestamp('2034-08-16'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
def w(f):
 d={}
 for a in A:
  x=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); d[a]=x.loc[:END,f]
 return pd.DataFrame(d)
o,c,h,l=map(w,['open','close','high','low']); r=c.pct_change(fill_method=None)
bp=((c-o)/(h-l).replace(0,np.nan)).clip(-1,1)
# Signed persistence: recent share of emphatically positive body sessions minus emphatically negative sessions.
raw=(bp>.65).rolling(20,min_periods=15).mean()-(bp<-.65).rolling(20,min_periods=15).mean()
trend=(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std()
s=pd.DataFrame(index=c.index,columns=A,dtype=float)
for t in c.index:
 q=pd.concat([raw.loc[t],trend.loc[t]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,1].std()>1e-12:
  b=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/np.var(q.iloc[:,1],ddof=1); s.loc[t,q.index]=q.iloc[:,0]-(q.iloc[:,0].mean()-b*q.iloc[:,1].mean())-b*q.iloc[:,1]
print('IDEA trend_orthogonal_extreme_intraday_body_persistence_20obs','endpoint',END.date(),'rows',len(c),'assets',len(A),'cells',int(s.notna().sum().sum()),'coverage',round(s.notna().mean().mean(),6))
for H in [1,5,10,20]:
 y=c.shift(-H)/c-1; z=[]; ds=[];ns=[]
 for t in c.index:
  q=pd.concat([s.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds); print('H',H,'dates',len(z),'meanN',round(np.mean(ns),2),'minN',min(ns),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),5))
 for nm,mask in [('early',ds<'2025-01-01'),('mid',(ds>='2025-01-01')&(ds<'2030-01-01')),('late',ds>='2030-01-01')]:
  zz=z[mask];print(' regime',nm,'n',len(zz),'IC',round(zz.mean(),6) if len(zz) else None,'ICIR',round(zz.mean()/zz.std(ddof=1),6) if len(zz)>1 else None)
rk=s.rank(axis=1,pct=True); d=rk.diff().abs();print('TURNOVER',round(d.stack().mean(),6),'comparisons',int(d.notna().sum().sum()),'CONCENTRATION_MEDIAN_IQR',round(s.quantile(.75,axis=1).sub(s.quantile(.25,axis=1)).median(),8))
# Close-location proxy establishes whether thresholded body-event persistence merely restates location pressure.
clv=((c-l)/(h-l).replace(0,np.nan)).clip(0,1); clraw=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean(); cs=pd.DataFrame(index=c.index,columns=A,dtype=float)
for t in c.index:
 q=pd.concat([clraw.loc[t],trend.loc[t]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,1].std()>1e-12:
  b=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/np.var(q.iloc[:,1],ddof=1);cs.loc[t,q.index]=q.iloc[:,0]-(q.iloc[:,0].mean()-b*q.iloc[:,1].mean())-b*q.iloc[:,1]
for nm,x in [('trend',trend),('trend_orthogonal_close_location_pressure',cs)]:
 q=pd.concat([s.stack(),x.stack()],axis=1).dropna();print('NOVELTY_PROXY',nm,'cells',len(q),'rho',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6))
