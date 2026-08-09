"""One-idea validation: participation-adaptive downside-recovery residual, cutoff 2030-01-09."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2030-01-09')
def rd(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(axis=1)
v=R.rolling(20,min_periods=15).std(); mom=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def res(x,*cs):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
def beta(mask):
 m=M.where(mask); return pd.DataFrame({a:R[a].where(mask).rolling(30,min_periods=10).cov(m)/m.rolling(30,min_periods=10).var() for a in A})
dba=-(beta(M<0)-beta(M>0)); V=pd.DataFrame({a:rd(a,'volume') for a in A}); rel=V/V.rolling(20,min_periods=15).mean()
# Prior-loss recovery is volume-confirmed when volume has a reliable history; otherwise the
# nonparticipation fallback avoids systematic exclusion of valid yield instruments.
avail=V.rolling(20,min_periods=1).count()>=15
raw=R.where((R.shift(1)<0)&((rel>0.9)|(~avail))).rolling(20,min_periods=4).mean()/(v+1e-12)
F=res(raw,v,peer,dba,mom)
print('FACTOR participation_adaptive_downside_recovery_residual_20 visible_through',E.date(),'assets',len(A),'raw_cells',int(raw.notna().sum().sum()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append((t,q.f.corr(q.r,method='spearman'))); ns.append(len(q))
 x=pd.Series(dict(vals)); ics[h]=x; sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,mask in [('2026_27',(ics[10].index>='2026-01-01')&(ics[10].index<'2028-01-01')),('2028_29',ics[10].index>='2028-01-01')]:
 x=ics[10][mask]; print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); tos=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8: tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(tos):.6f}')
# Closest admitted recovery construction reconstructed for a conservative novelty diagnostic.
sev=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,5); cont=R.mul(sev).rolling(20,min_periods=10).sum().div(sev.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
loss=(R<0).astype(float); lc=pd.DataFrame({a:loss[a].rolling(20,min_periods=15).cov(loss[a].shift())/(loss[a].rolling(20,min_periods=15).mean()*(1-loss[a].rolling(20,min_periods=15).mean())) for a in A})
existing=res(res(res(cont,R.where(R.shift(1)<0).rolling(20,min_periods=6).mean()/(v+1e-12)),lc),mom)
q=pd.concat([F.stack(),existing.stack()],axis=1).dropna(); print('closest_reconstructed_library_rho',f'{q.iloc[:,0].corr(q.iloc[:,1],method="spearman"):.6f}','common_cells',len(q))
