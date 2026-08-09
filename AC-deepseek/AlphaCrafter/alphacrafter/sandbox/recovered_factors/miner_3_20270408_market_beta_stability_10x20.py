"""miner_3: test cross-asset market-beta stability (one interpretable fragility idea)."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2027-04-07')
def close(path):
 d=pd.read_csv(path,parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A});R=P.pct_change(fill_method=None);M=R.mean(axis=1)
# Candidate: stable market exposure: negative 20d standard deviation of a trailing 10d rolling beta to equal-weight market.
B=pd.DataFrame({a:R[a].rolling(10,min_periods=7).cov(M)/M.rolling(10,min_periods=7).var() for a in A})
F=-B.rolling(20,min_periods=15).std()
# reconstruct active library signals exactly enough for binding pooled Spearman comparison
v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std()
def resid(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>0:
   b=z.x.cov(z.y)/z.y.var();o.loc[dt,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
peer=pd.DataFrame(index=P.index,columns=A)
for a in A: peer[a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1)
def macrobeta(fn):
 m=close('../persistent/index_data/'+fn).reindex(P.index).pct_change(fill_method=None)
 return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),peer)
v=close('../persistent/index_data/VIX.csv').reindex(P.index); shock=(v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0)
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
def cb(a,mask):
 x=M.where(mask);y=R[a].where(mask);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
L={'ravmom':(P/P.shift(20)-1)/v20,'reversal':-(P/P.shift(5)-1)/v5,'realized_vol':v20,'peer':peer,'vix_resid':macrobeta('VIX.csv'),'dxy_resid':macrobeta('DXY.csv'),'high_vix_asym':resid(np.log((up+1e-8)/(dn+1e-8)),P/P.shift(20)-1).mul(shock,axis=0),'vix_peer':peer.mul(shock,axis=0),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'compression':-(v5/(v60+1e-12)),'downside_beta_asym':pd.DataFrame({a:-(cb(a,M<0)-cb(a,M>0)) for a in A})}
print('FACTOR market_beta_stability_10x20 = -rolling_std_20(rolling_beta_10(asset,equal_weight_market)); visible through',END.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;vals=[];ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x;sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
x=ics[10]
for n,mask in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
 q=x[mask];print(n,'dates',len(q),'IC',None if len(q)==0 else f'{q.mean():.6f}','ICIR',None if len(q)==0 else f'{q.mean()/q.std(ddof=1):.6f}','hit',None if len(q)==0 else f'{(q>0).mean():.4f}')
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('signal_cell_coverage',f'{F.notna().mean().mean():.6f}','mean_daily_rank_turnover',f'{np.mean(turn):.6f}')
mx=-1;who='';cells=0
for n,s in L.items():
 z=pd.concat([F.stack().rename('f'),s.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','common_cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'active_library_records',len(glob.glob('factors/*.json')))
