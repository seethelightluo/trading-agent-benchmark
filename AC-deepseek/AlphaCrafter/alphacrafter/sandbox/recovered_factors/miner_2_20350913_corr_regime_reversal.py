import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-09-12')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None)
# average pairwise rolling correlation, requiring a broad panel
corr=[]
for t in R.index:
 z=R.loc[:t].tail(20).dropna(axis=1,how='any')
 corr.append(z.corr().where(~np.eye(len(z.columns),dtype=bool)).stack().mean() if len(z.columns)>=8 else np.nan)
co=pd.Series(corr,index=R.index)
# elevated common-correlation regime; lagged and bounded to avoid scale domination
cm=co.rolling(60,min_periods=30).rank(pct=True).shift(1)
vol=R.rolling(5,min_periods=4).std(); raw=-(P/P.shift(5)-1)/(vol+1e-12)
F=raw.mul((0.5+cm),axis=0).shift(1)
F=F.sub(F.mean(axis=1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=correlation-regime-conditioned vol-normalized reversal rows',len(P),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',F.notna().mean().mean())
def test(h):
 fw=P.shift(-h)/P-1; out=[]; dates=[]; nn=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1: out.append(q.f.corr(q.r,method='spearman')); dates.append(t); nn.append(len(q))
 x=pd.Series(out,index=dates); print('H',h,'dates',len(x),'meanN',round(np.mean(nn),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4)); return x
X={h:test(h) for h in [1,5,10,20]}
# transparent proxy library audit; exact admitted signal reconstruction unavailable in this cycle
L={'reversal_5':-(P/P.shift(5)-1),'momentum_20':P/P.shift(20)-1,'volatility_20':R.rolling(20,min_periods=15).std(),'corr_regime':pd.DataFrame({a:cm for a in A})}
mx=0;who='';cells=0
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); r=q.f.corr(q.x,method='spearman'); print('proxy',n,'rho',round(r,6),'cells',len(q))
 if abs(r)>mx:mx=abs(r);who=n;cells=len(q)
print('max_abs_library_correlation',round(mx,6),'against',who,'common_cells',cells,'NOTE exact 30-factor audit required for admission')
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-09-12')]:
 x=X[5][(X[5].index>=lo)&(X[5].index<=hi)]; print('regime',lo,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
