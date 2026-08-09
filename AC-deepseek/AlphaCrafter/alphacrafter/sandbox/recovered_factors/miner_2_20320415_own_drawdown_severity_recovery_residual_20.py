"""miner_2: asset-specific drawdown-severity-conditioned recovery residual, one idea."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2032-04-14')
def series(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:series(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(1)
vol=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1; loss=(R<0).rolling(20,min_periods=15).mean()
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
# Severity known at t-1: each asset's lagged 20d peak drawdown, normalized to [0,1].
dd=P/P.rolling(20,min_periods=15).max()-1
severity=(-dd.shift(1)).clip(0,1)
# Average volatility-scaled returns observed after deeper own drawdown states in trailing 20 sessions.
raw=R.div(vol+1e-12).mul(severity).rolling(20,min_periods=12).sum().div(severity.rolling(20,min_periods=12).sum().replace(0,np.nan))
def residual(x,controls):
 out=pd.DataFrame(np.nan,index=P.index,columns=A)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(controls)],axis=1).dropna()
  X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:
   out.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return out
F=residual(raw,[vol,trend,loss,peer,dd])
print('FACTOR own_drawdown_severity_recovery_residual_20 endpoint',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.r,method='spearman')
   if np.isfinite(z):vals.append((t,z));ns.append(len(q))
 x=pd.Series(dict(vals)); ics[h]=x
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for label,mask in [('2026_2027',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_current',ics[20].index>='2028-01-01')]:
 x=ics[20][mask];print('regime20',label,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
ranks=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'mean_rank_turnover',f'{np.mean(ts):.6f}')
