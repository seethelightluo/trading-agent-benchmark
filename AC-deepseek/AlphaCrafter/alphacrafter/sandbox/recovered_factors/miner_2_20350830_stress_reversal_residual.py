import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-08-29')
def rd(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); return pd.to_numeric(d.loc[:E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); breadth=(R>0).sum(axis=1)/R.notna().sum(axis=1); vol5=R.rolling(5,min_periods=4).std(); rev=-(P/P.shift(5)-1)/(vol5+1e-12); stress=(1-breadth).rolling(10,min_periods=8).mean().shift(1)
raw=rev.mul(stress,axis=0).shift(1); base=rev.shift(1)
def resid(row,b):
 q=pd.concat([row.rename('x'),b.rename('b')],axis=1).dropna()
 if len(q)<8 or q.b.nunique()<2:return row* np.nan
 X=np.column_stack([np.ones(len(q)),q.b.values]); beta=np.linalg.lstsq(X,q.x.values,rcond=None)[0]; out=row.copy();out.loc[q.index]=q.x.values-X@beta;return out
F=pd.DataFrame([resid(raw.loc[t],base.loc[t]) for t in raw.index],index=raw.index); F=F.sub(F.mean(axis=1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=stress reversal residualized against standard reversal','valid',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
def test(h):
 fw=P.shift(-h)/P-1;out=[];ds=[];nn=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1:out.append(q.f.corr(q.r,method='spearman'));ds.append(t);nn.append(len(q))
 x=pd.Series(out,index=ds);print('H',h,'dates',len(x),'N',np.mean(nn),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean());return x
X={h:test(h) for h in [1,5,10,20]}; q=pd.concat([F.stack(),base.stack()],axis=1).dropna();print('corr_standard_reversal',q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-08-29')]:
 x=X[10][(X[10].index>=lo)&(X[10].index<=hi)];print('regime',lo,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
