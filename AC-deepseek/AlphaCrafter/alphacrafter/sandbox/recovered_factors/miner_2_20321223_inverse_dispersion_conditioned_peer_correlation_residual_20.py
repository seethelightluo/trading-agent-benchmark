"""Miner_2: validate one idea: inverse high-dispersion peer-correlation residual."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2032-12-22')
def rd(a,root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(axis=1)
v=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
# Candidate: when cross-asset dispersion is unusually high, low peer correlation is
# interpreted as independent price discovery / diversification resilience. The negative
# conditional correlation is residualized so it is not merely low volatility, trend,
# or unconditional crowding.
disp=R.std(axis=1); z=(disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)
raw=-peer*z.clip(lower=0)
def resid(y, controls):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([y.loc[t].rename('y')]+[x.loc[t].rename(str(i)) for i,x in enumerate(controls)],axis=1).dropna()
  X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:
   out.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return out
# Controls deliberately isolate conditional diversification from normal crowding,
# 20d risk, and directionality.
F=resid(raw,[peer,v,trend])
print('FACTOR inverse_dispersion_conditioned_peer_correlation_residual_20 visible_through',E.date(),'assets',len(A))
ics={}
for h in (1,5,10,20):
 fw=P.shift(-h)/P-1; rows=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   rows.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(rows));ics[h]=x
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for label,mask in [('2026_27',(ics[10].index>='2026-01-01')&(ics[10].index<'2028-01-01')),('2028_30',(ics[10].index>='2028-01-01')&(ics[10].index<'2031-01-01')),('2031_current',ics[10].index>='2031-01-01')]:
 x=ics[10][mask]; print('regime10',label,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); turn=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(turn):.6f}')
# Novelty diagnostics against closest established signal families; exact full-library
# screening is only required if this candidate passes paper IC gates.
for n,x in {'peer_crowding':peer,'realized_vol':v,'ravmom':trend/(v+1e-12),'dispersion_weighted_peer':raw}.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();print('proxy_rho',n,f'{q.f.corr(q.x,method="spearman"):.6f}','cells',len(q))
