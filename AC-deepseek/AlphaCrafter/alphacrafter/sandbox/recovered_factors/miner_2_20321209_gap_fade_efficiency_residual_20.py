# Point-in-time validation of one candidate: Gap-Fade Efficiency Residual (20)
# Uses only through 2032-12-08; forward returns are per-asset observed-session shifts.
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2032-12-08')
def rd(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(axis=1)
v=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); L=pd.DataFrame({a:rd(a,'low') for a in A})
loc=((P-L)/(H-L).replace(0,np.nan)).clip(0,1); gap=O/P.shift(1)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def bet(mask):
 m=M.where(mask);return pd.DataFrame({a:R[a].where(mask).rolling(30,min_periods=10).cov(m)/m.rolling(30,min_periods=10).var() for a in A})
dba=-(bet(M<0)-bet(M>0)); cl=loc.where(R<0).rolling(20,min_periods=6).mean()-loc.where(R>0).rolling(20,min_periods=6).mean()
raw=(-gap*(2*loc-1)).rolling(20,min_periods=12).mean()/(v+1e-12)
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in P.index:
 q=pd.concat([raw.loc[t].rename('y'),v.loc[t].rename('v'),peer.loc[t].rename('p'),dba.loc[t].rename('b'),trend.loc[t].rename('t'),cl.loc[t].rename('c')],axis=1).dropna()
 X=np.c_[np.ones(len(q)),q.iloc[:,1:].to_numpy()]
 if len(q)>=8 and np.linalg.matrix_rank(X)==X.shape[1]:F.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
print('FACTOR gap_fade_efficiency_residual_20 cutoff',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=pd.DataFrame({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index) for a in A}); out=[];nn=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(out));ics[h]=x; sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',(ics[10].index>='2026-01-01')&(ics[10].index<'2028-01-01')),('2028_30',(ics[10].index>='2028-01-01')&(ics[10].index<'2030-01-01')),('2030_32',ics[10].index>='2030-01-01')]:
 x=ics[10][mask]; print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
