"""miner_3 one idea: VIX-shock-amplified residual 5d reversal."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2026-12-30');P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();P[a]=pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame(P);R=P.pct_change(fill_method=None); raw=-(P/P.shift(5)-1);base=-((P/P.shift(5)-1)/R.rolling(5,min_periods=4).std());F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 z=pd.concat([raw.loc[dt].rename('x'),base.loc[dt].rename('b')],axis=1).dropna()
 if len(z)>=8 and z.b.var()>0:
  b=np.cov(z.x,z.b,ddof=1)[0,1]/z.b.var();F.loc[dt,z.index]=z.x-(z.x.mean()-b*z.b.mean()+b*z.b)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index().close.astype(float).reindex(P.index);shock=(v/v.shift(5)-1).clip(lower=0);F=F.mul(1+shock,axis=0)
# exact/reasonable reconstructions of seven admitted signals, including incumbent asymmetry
vol=R.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A});L={'ravmom':(P/P.shift(20)-1)/vol,'volnorm_reversal':base,'realized_vol':vol,'peer_crowding':peer}
def macro(fn):
 m=pd.read_csv('../persistent/index_data/'+fn,parse_dates=['date']).query('date<=@END').set_index('date').close.astype(float).pct_change(fill_method=None).reindex(P.index);x=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A})
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),peer.loc[dt].rename('p')],axis=1).dropna()
  if len(z)>=8 and z.p.var()>0:
   b=np.cov(z.x,z.p,ddof=1)[0,1]/z.p.var();x.loc[dt,z.index]=z.x-(z.x.mean()-b*z.p.mean()+b*z.p)
 return x
L['vix_beta_resid']=macro('VIX.csv');L['dxy_beta_resid']=macro('DXY.csv');up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);aa=np.log((up+1e-8)/(dn+1e-8));mom=P/P.shift(20)-1;res=pd.DataFrame(index=P.index,columns=A)
for dt in P.index:
 z=pd.concat([aa.loc[dt].rename('x'),mom.loc[dt].rename('m')],axis=1).dropna()
 if len(z)>=8 and z.m.var()>0:
  b=np.cov(z.x,z.m,ddof=1)[0,1]/z.m.var();res.loc[dt,z.index]=z.x-(z.x.mean()-b*z.m.mean()+b*z.m)
L['high_vix_asym']=res.mul((v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
def stat(x):return x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),x.std(ddof=1)/np.sqrt(len(x))
def ic(h):
 fw=P.shift(-h)/P-1;out=[];nn=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:out.append((dt,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 return pd.Series(dict(out)),np.mean(nn)/15
print('FACTOR vix_shock_amplified_residual_reversal_5; visible through',END.date(),'instruments=15');print('expression: (1+max(VIX/VIX[-5]-1,0))*residual_cs[-return5 ~ -return5/vol5]')
for h in [1,5,10,20]:
 x,c=ic(h);a,b,hit,se=stat(x);print(f'h={h} dates={len(x)} IC={a:.6f} ICIR={b:.6f} hit={hit:.4f} se={se:.6f} coverage={c:.4f} mean_n={c*15:.2f}')
 if h==5:
  for n,q in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026',x.index>='2026-01-01')]:
   y=x[q];print(n,'dates',len(y),f'IC={y.mean():.6f}',f'ICIR={y.mean()/y.std(ddof=1):.6f}',f'hit={(y>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',np.mean(to),'signal_cell_coverage',F.notna().mean().mean(),'positive_shock_days',(shock>0).sum(),'/',shock.notna().sum())
mx=0
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');mx=max(mx,abs(rho));print('library',n,f'rho={rho:.6f}','cells',len(z))
print('max_abs_library_correlation',f'{mx:.6f}','library_json_records',len([x for x in glob.glob('factors/*.json') if not x.endswith('.bak')]))
