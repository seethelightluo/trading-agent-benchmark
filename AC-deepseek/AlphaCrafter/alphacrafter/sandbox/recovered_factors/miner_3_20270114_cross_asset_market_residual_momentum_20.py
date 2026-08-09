"""miner_3 one idea: 20-observation cross-asset-market residual momentum."""
import numpy as np, pandas as pd, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-01-13')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 P[a]=pd.to_numeric(d.loc[d.index<=END,'close'],errors='coerce')
P=pd.DataFrame(P).sort_index(); R=P.pct_change(fill_method=None)
# A common daily cross-asset move is observable at each date. Estimate each asset's 60d beta,
# then retain its cumulative 20d return unexplained by that common move.
M=R.mean(axis=1,skipna=True)
beta=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A: beta[a]=R[a].rolling(60,min_periods=40).cov(M)/M.rolling(60,min_periods=40).var()
F=(P/P.shift(20)-1).sub(beta.mul(M.rolling(20,min_periods=15).sum(),axis=0))
# reconstructed admitted-library signals for mandatory signal-correlation evidence
vol=R.rolling(20,min_periods=15).std(); base=-((P/P.shift(5)-1)/R.rolling(5,min_periods=4).std())
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
L={'ravmom':(P/P.shift(20)-1)/vol,'volnorm_reversal':base,'realized_vol':vol,'peer_crowding':peer}
def residual(x, against):
 out=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),against.loc[dt].rename('b')],axis=1).dropna()
  if len(z)>=8 and z.b.var()>0:
   k=np.cov(z.x,z.b,ddof=1)[0,1]/z.b.var(); out.loc[dt,z.index]=z.x-(z.x.mean()-k*z.b.mean()+k*z.b)
 return out
def macro(name):
 q=pd.read_csv('../persistent/index_data/'+name,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
 q=q.loc[q.index<=END].pct_change(fill_method=None).reindex(P.index)
 return residual(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(q) for a in A}),peer)
L['vix_beta_resid']=macro('VIX.csv'); L['dxy_beta_resid']=macro('DXY.csv')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float).loc[lambda x:x.index<=END].reindex(P.index)
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
L['high_vix_asym']=residual(np.log((up+1e-8)/(dn+1e-8)),P/P.shift(20)-1).mul((v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
def getic(h):
 fw=P.shift(-h)/P-1; x=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: x.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 return pd.Series(dict(x)),np.mean(ns)/15
print('FACTOR cross_asset_market_residual_momentum_20; visible through',END.date(),'instruments=15')
print('expression: return_20(asset) - beta_60(asset, equal_weight_daily_return)*sum_20(equal_weight_daily_return)')
for h in [1,5,10,20]:
 x,c=getic(h); sd=x.std(ddof=1); print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f} mean_n={c*15:.2f}')
 if h==5:
  for n,q in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
   y=x[q];print(n,'dates',len(y),f'IC={y.mean():.6f}',f'ICIR={y.mean()/y.std(ddof=1):.6f}',f'hit={(y>0).mean():.4f}')
r=F.rank(axis=1,pct=True); tos=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',np.mean(tos),'signal_cell_coverage',F.notna().mean().mean())
mx=0
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');mx=max(mx,abs(rho));print('library',n,f'rho={rho:.6f}','cells',len(z))
print('max_abs_library_correlation',f'{mx:.6f}','library_json_records',len(glob.glob('factors/*.json')))
