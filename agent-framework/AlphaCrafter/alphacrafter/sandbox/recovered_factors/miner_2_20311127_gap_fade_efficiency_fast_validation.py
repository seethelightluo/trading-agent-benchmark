"""Single-idea exploratory validation: 20-session gap-fade efficiency, point-in-time through 2031-11-26."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2031-11-26')
def get(a,c):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:get(a,'close') for a in A}); O=pd.DataFrame({a:get(a,'open') for a in A}); H=pd.DataFrame({a:get(a,'high') for a in A}); L=pd.DataFrame({a:get(a,'low') for a in A})
R=P.pct_change(fill_method=None); vol=R.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:R.drop(columns=a).rolling(20,min_periods=15).corr(R[a]).mean(axis=1) for a in A})
loc=((P-L)/(H-L).replace(0,np.nan)).clip(0,1); gap=O/P.shift(1)-1; raw=(-gap*(2*loc-1)).rolling(20,min_periods=12).mean()/(vol+1e-12)
trend=P/P.shift(20)-1; cl=loc.where(R<0).rolling(20,min_periods=6).mean()-loc.where(R>0).rolling(20,min_periods=6).mean()
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in P.index:
 q=pd.concat([raw.loc[t].rename('y'),vol.loc[t].rename('v'),peer.loc[t].rename('p'),trend.loc[t].rename('m'),cl.loc[t].rename('c')],axis=1).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q.iloc[:,1:]]
  if np.linalg.matrix_rank(X)==X.shape[1]: F.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
print('FACTOR gap_fade_efficiency_residual_20 visible_through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 z=[]; ns=[]; fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t],fw.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 x=pd.Series(dict(z)); ics[h]=x; print('h',h,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}','instruments',f'{np.mean(ns):.2f}')
x=ics[20]
for n,m in [('2020_23',x.index<'2024-01-01'),('2024_27',(x.index>='2024-01-01')&(x.index<'2028-01-01')),('2028_31',x.index>='2028-01-01')]:
 y=x[m];print('regime',n,'dates',len(y),'IC',f'{y.mean():.6f}','ICIR',f'{y.mean()/y.std(ddof=1):.6f}','hit',f'{(y>0).mean():.4f}')
r=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(turns):.6f}')
print('NOVELTY NOT COMPUTED: full admitted-library comparison required; no admission possible without it.')
