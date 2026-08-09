"""Single-idea diagnostic: validate negative-gap intraday recovery residual using stored full history, signals through 2030-04-03."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2030-04-03')
def load(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d[c],errors='coerce')
P=pd.DataFrame({a:load(a) for a in A}); O=pd.DataFrame({a:load(a,'open') for a in A})
R=P.pct_change(fill_method=None); M=R.mean(axis=1); V=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if a!=b],axis=1).mean(axis=1) for a in A})
def beta(mask):
 m=M.where(mask); return pd.DataFrame({a:R[a].where(mask).rolling(30,min_periods=10).cov(m)/m.rolling(30,min_periods=10).var() for a in A})
dba=-(beta(M<0)-beta(M>0))
def residual(y, controls):
 z=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([y.loc[t].rename('y')]+[c.loc[t].rename(str(i)) for i,c in enumerate(controls)],axis=1).dropna()
  if len(q)>=8:
   X=np.c_[np.ones(len(q)),q.iloc[:,1:].to_numpy()]
   if np.linalg.matrix_rank(X)==X.shape[1]: z.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return z
# Signal at t uses data through t: severity-weighted same-day close/open recovery after adverse opening gap.
gap=O/P.shift(1)-1; intr=P/O-1; sev=(-gap/(V.shift(1)+1e-12)).clip(lower=0,upper=4)
raw=intr.mul(sev).rolling(20,min_periods=10).sum().div(sev.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0).div(V+1e-12)
F=residual(raw,[V,peer,dba,trend])
print('FACTOR negative_overnight_gap_intraday_recovery_residual_20 visible_through',E.date(),'assets',len(A),'price_dates',len(P),'raw_cells',int(raw.notna().sum().sum()),'factor_cells',int(F.notna().sum().sum()))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; out=[]; ns=[]
 for t in P.index[(P.index<=E)]:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: out.append((t,q.f.corr(q.r,method='spearman'))); ns.append(len(q))
 x=pd.Series({t:v for t,v in out},dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for name,lo,hi in [('2020_21','2020-01-01','2022-01-01'),('2022_23','2022-01-01','2024-01-01'),('2024_25','2024-01-01','2026-01-01'),('2026_27','2026-01-01','2028-01-01'),('2028_30','2028-01-01','2031-01-01')]:
 x=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)]; print('regime',name,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else 'nan','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); tos=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8: tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','turnover',f'{np.mean(tos):.6f}')
# Mandatory library evidence cannot be claimed from incomplete reconstructions: calculate robust correlation with existing stored factor files only where signal matrices are encoded; if unavailable, admission fails.
print('LIBRARY_NOVELTY_EVIDENCE unavailable: admitted JSON definitions contain no historical signal panels; candidate automatically fails novelty admission.')
