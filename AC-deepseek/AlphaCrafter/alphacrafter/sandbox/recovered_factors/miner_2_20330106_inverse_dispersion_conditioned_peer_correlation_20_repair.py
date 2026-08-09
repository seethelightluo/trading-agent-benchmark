"""Repair validation, one idea: inverse dispersion-conditioned peer correlation (20).
Uses shared observation dates only; all prices and forward shifts are point-in-time per asset."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-01-05')
def read(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A}).sort_index(); R=P.pct_change(fill_method=None)
# Pairwise corr preserves each pair's common sessions.  Its result is then explicitly reindexed to P dates.
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]).rename(b) for b in A if b!=a],axis=1).mean(axis=1).reindex(P.index) for a in A})
disp=R.std(axis=1,skipna=True); z=(disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)
F=peer.mul(-z.clip(lower=0),axis=0)
print('FACTOR inverse_dispersion_conditioned_peer_correlation_20 repaired cutoff',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=pd.DataFrame({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index) for a in A}); vals=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series([v for _,v in vals],index=pd.DatetimeIndex([t for t,_ in vals]),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,mask in [('2020_22',ics[10].index<pd.Timestamp('2023-01-01')),('2023_25',(ics[10].index>=pd.Timestamp('2023-01-01'))&(ics[10].index<pd.Timestamp('2026-01-01'))),('2026_29',(ics[10].index>=pd.Timestamp('2026-01-01'))&(ics[10].index<pd.Timestamp('2030-01-01'))),('2030_current',ics[10].index>=pd.Timestamp('2030-01-01'))]:
 x=ics[10][mask];print('regime10',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);tos=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(tos):.6f}')
# Diagnostic similarity only, not library-admission evidence.
for n,X in [('peer',peer),('dispersion_condition',pd.DataFrame({a:z for a in A}))]:
 q=pd.concat([F.stack().rename('f'),X.stack().rename('x')],axis=1).dropna();print('diagnostic_rho',n,f'{q.f.corr(q.x,method="spearman"):.6f}','cells',len(q))
