"""miner_3: downside correlation resilience, one conditional cross-asset diversification factor."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-05-19')
def load(p,col='close'):
 d=pd.read_csv(p,parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[col],errors='coerce')
P=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A});R=P.pct_change(fill_method=None);M=R.mean(axis=1)
def avgcorr(mask=None):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for a in A:o[a]=pd.concat([R[a].where(mask).rolling(40,min_periods=15).corr(R[b].where(mask)) for b in A if b!=a],axis=1).mean(axis=1)
 return o
# High rank: correlation to peers falls, rather than rises, in broad down-market days relative to unconditional 40d correlation.
U=avgcorr();D=avgcorr(M<0);F=U-D
v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std(); peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def resid(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>0:
   b=z.x.cov(z.y)/z.y.var();o.loc[dt,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
def mb(fn):
 m=load('../persistent/index_data/'+fn).reindex(P.index).pct_change(fill_method=None);return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),peer)
v=load('../persistent/index_data/VIX.csv').reindex(P.index);sh=(v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0);up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
def cb(a,mask):return R[a].where(mask).rolling(30,min_periods=10).cov(M.where(mask))/M.where(mask).rolling(30,min_periods=10).var()
ind=(R<0).astype(float);loss=ind.rolling(20,min_periods=12).cov(ind.shift(1))/(ind.rolling(20,min_periods=12).mean()*(1-ind.rolling(20,min_periods=12).mean())+1e-12)
L={'ravmom':(P/P.shift(20)-1)/v20,'reversal':-(P/P.shift(5)-1)/v5,'realized_vol':v20,'peer_crowding':peer,'vix_beta_resid':mb('VIX.csv'),'dxy_beta_resid':mb('DXY.csv'),'vix_peer':peer.mul(sh,axis=0),'high_vix_asym':resid(np.log((up+1e-8)/(dn+1e-8)),P/P.shift(20)-1).mul(sh,axis=0),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'compression':-v5/(v60+1e-12),'downside_beta_asym':pd.DataFrame({a:-(cb(a,M<0)-cb(a,M>0)) for a in A}),'loss_clustering':loss,'close_resilience':None,'inverse_rav_resid':None}
print('FACTOR downside_correlation_resilience_40 = unconditional 40d mean peer correlation minus mean peer correlation conditional on equal-weight cross-asset market<0; visible through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:out.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
x=ics[10]
for n,m in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
 q=x[m];print(n,'dates',len(q),'IC',f'{q.mean():.6f}' if len(q) else None,'ICIR',f'{q.mean()/q.std(ddof=1):.6f}' if len(q)>1 else None,'hit',f'{(q>0).mean():.4f}' if len(q) else None)
r=F.rank(axis=1,pct=True);t=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('signal_cell_coverage',f'{F.notna().mean().mean():.6f}','mean_daily_rank_turnover',f'{np.mean(t):.6f}')
# exact persisted definitions requiring special signals omitted only reported as failed evidence; broad proxies cover shared construction family
mx=0;who='';cells=0
for n,s in L.items():
 if s is None: print('library',n,'MISSING_EVIDENCE');continue
 z=pd.concat([F.stack().rename('f'),s.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','common_cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation_INCOMPLETE',f'{mx:.6f}','against',who,'common_cells',cells,'active_library_records',len(glob.glob('factors/*.json')))
