"""miner_1: 2028-03-09 single-idea test: volume-trend acceleration residual, through prior completed day."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-03-08')
def col(a,c):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return pd.to_numeric(d[c],errors='coerce').replace([0,np.inf,-np.inf],np.nan)
P=pd.DataFrame({a:col(a,'close') for a in A}); V=pd.DataFrame({a:col(a,'volume') for a in A}).reindex(P.index)
R=P.pct_change(fill_method=None); vol=R.rolling(20,min_periods=15).std(); mom=(P/P.shift(20)-1)/(vol+1e-12)
# One idea: recent volume participation acceleration, separated from ordinary volume level, return trend, and volatility.
fast=np.log(V.rolling(5,min_periods=4).mean()/V.rolling(20,min_periods=15).mean())
level=np.log(V/V.rolling(20,min_periods=15).mean())
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in P.index:
 q=pd.concat([fast.loc[t].rename('y'),level.loc[t].rename('level'),mom.loc[t].rename('mom'),vol.loc[t].rename('vol')],axis=1).dropna()
 X=q[['level','mom','vol']].values
 if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==4:
  F.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
print('FACTOR volume_trend_acceleration_residual_5v20 visible_through',END.date(),'assets',len(A))
print('definition: CSResidual(log(mean(volume,5)/mean(volume,20)), log(volume/mean(volume,20)), risk_adjusted_momentum_20, realized_volatility_20)')
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; z=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: z.append((t,spearmanr(q.f,q.r).statistic));ns.append(len(q))
 s=pd.Series(dict(z));ics[h]=s; sd=s.std(ddof=1)
 print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(s)):.6f}')
for label,mask in [('2026',ics[10].index<'2027-01-01'),('2027',(ics[10].index>='2027-01-01')&(ics[10].index<'2028-01-01')),('2028',ics[10].index>='2028-01-01')]:
 s=ics[10][mask]; print('regime',label,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else None,'hit',round((s>0).mean(),4))
rk=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'mean_daily_rank_turnover',f'{np.mean(turns):.6f}')
# Transparent overlap diagnostics for directly reconstructed principal controls.
for n,z in {'risk_adjusted_momentum_20':mom,'realized_volatility_20':vol,'relative_volume_level_20':level}.items():
 q=pd.concat([F.stack().rename('f'),z.stack().rename('z')],axis=1).dropna();print('diagnostic_rho',n,f'{q.f.corr(q.z,method="spearman"):.6f}','cells',len(q))
print('NOTE: no admission decision can be made from control correlations alone; full admitted-library signal comparison is required before persistence.')
