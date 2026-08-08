import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-09-26')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None)
# Cross-asset residual downside-rebound: recent residual return versus equal-weight benchmark,
# with only downside shocks amplified; lagged to prevent lookahead.
M=R.mean(axis=1); beta=R.rolling(40,min_periods=25).cov(M).div(M.rolling(40,min_periods=25).var(),axis=0)
res=R.sub(beta.mul(M,axis=0),axis=1)
down=(res.clip(upper=0)).rolling(10,min_periods=6).sum()
res10=res.rolling(10,min_periods=7).sum()
vol=R.rolling(20,min_periods=12).std()
F=(-(res10+0.75*down)/(vol+1e-12)).shift(1)
F=F.sub(F.mean(axis=1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=downside-shock residual rebound rows',len(P),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',round(F.notna().mean().mean(),4))
def test(h):
 fw=P.shift(-h)/P-1; out=[]; dates=[]; nn=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1: out.append(q.f.corr(q.r,method='spearman')); dates.append(t); nn.append(len(q))
 x=pd.Series(out,index=dates); print('H',h,'dates',len(x),'meanN',round(np.mean(nn),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4)); return x
X={h:test(h) for h in [1,5,10,20]}
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-09-26')]:
 x=X[10][(X[10].index>=lo)&(X[10].index<=hi)]; print('regime',lo,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# broad proxies, not library proof
for n,x in {'residual10':res10,'downside':down,'vol20':vol}.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('proxy',n,'rho',round(q.f.corr(q.x,method='spearman'),6),'cells',len(q))
print('LIBRARY_AUDIT=FAILED exact aligned 30-factor signal histories unavailable; no admission without max_abs_library_correlation')
