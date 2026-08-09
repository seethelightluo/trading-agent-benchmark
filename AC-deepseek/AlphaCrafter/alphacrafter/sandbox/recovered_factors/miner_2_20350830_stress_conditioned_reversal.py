import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-08-29')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); breadth=(R>0).sum(1)/R.notna().sum(1)
# Candidate: short-term reversal, amplified only when cross-asset breadth is weak/strong, with continuous nonlinearity.
# Contrarian signal is positive after losses; stress-conditioned multiplier rewards rebound when breadth is depressed.
v=R.rolling(20,min_periods=15).std(); rev=-(P/P.shift(5)-1)/(R.rolling(5,min_periods=4).std()+1e-12)
stress=(1-breadth).rolling(10,min_periods=8).mean().shift(1)
F=(rev*stress).shift(1)
F=F.sub(F.mean(1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=stress-conditioned volatility-normalized 5d reversal rows',len(P),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
def test(h):
 fw=P.shift(-h)/P-1; out=[]; nn=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1: out.append(q.f.corr(q.r,method='spearman'));nn.append(len(q))
 x=pd.Series(out,index=[F.index[i] for i in range(len(out))]); print('H',h,'dates',len(x),'meanN',np.mean(nn),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean()); return x
X={h:test(h) for h in [1,5,10,20]}
# broad controls for novelty screen
L={'volatility_20':v,'momentum_20':P/P.shift(20)-1,'reversal_5':-(P/P.shift(5)-1),'breadth':pd.DataFrame({a:breadth for a in A})}
for n in ['DXY','VIX','USDJPY','EURUSD']:
 z=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:E,'close'].pct_change().reindex(P.index); L[n+'_beta']=pd.DataFrame({a:R[a].rolling(30,min_periods=10).corr(z) for a in A})
mx=0;who='';cells=0
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); r=q.f.corr(q.x,method='spearman'); print('library',n,'rho',r,'cells',len(q))
 if abs(r)>mx:mx=abs(r);who=n;cells=len(q)
print('max_abs_library_correlation',mx,'against',who,'common_cells',cells)
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-08-29')]:
 x=X[5][(X[5].index>=lo)&(X[5].index<=hi)]; print('regime',lo,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
