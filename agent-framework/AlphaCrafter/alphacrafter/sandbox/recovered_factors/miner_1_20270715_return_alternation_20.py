import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
def fld(a,c):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[c],errors='coerce')
def idx(n):
 d=pd.read_csv('../persistent/index_data/'+n,parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame({a:fld(a,'close') for a in A}); R=P.pct_change(fill_method=None);M=R.mean(axis=1)
# Candidate: high signal is recent return alternation, interpreted as a cross-asset mean-reversion opportunity.
F=pd.DataFrame({a:-R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A})
v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std();mom=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def resid(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>0:
   b=z.x.cov(z.y)/z.y.var();o.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
def macro(n):
 q=idx(n).reindex(P.index).pct_change(fill_method=None);return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(q) for a in A}),peer)
vix=idx('VIX.csv').reindex(P.index);shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0);up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
def beta(a,mask):
 x=M.where(mask);y=R[a].where(mask);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
loc=pd.DataFrame({a:(fld(a,'close')-fld(a,'low'))/(fld(a,'high')-fld(a,'low')) for a in A}).reindex(P.index);downloc=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
neg=R.lt(0).astype(float);loss=pd.DataFrame({a:neg[a].rolling(20,min_periods=15).cov(neg[a].shift(1))/(neg[a].rolling(20,min_periods=15).mean()*(1-neg[a].rolling(20,min_periods=15).mean())) for a in A});skew=-R.rolling(20,min_periods=15).skew()
L={'ravmom':mom/v20,'reversal':-(P/P.shift(5)-1)/v5,'vix_resid':macro('VIX.csv'),'dxy_resid':macro('DXY.csv'),'realized_vol':v20,'peer':peer,'vix_peer':peer.mul(shock,axis=0),'loss_cluster':loss,'inverse_upside':-resid(up,mom),'inverse_skew':resid(resid(skew,mom),v20),'high_vix_asym':resid(np.log((up+1e-8)/(dn+1e-8)),mom).mul(shock,axis=0),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'compression':-v5/(v60+1e-12),'down_beta':pd.DataFrame({a:-(beta(a,M<0)-beta(a,M>0)) for a in A}),'down_loc':downloc}
print('FACTOR return_alternation_20 = -corr_20(r_t,r_t-1); cutoff',END.date(),'assets',len(A));ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: out.append((t,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));ics[h]=x;s=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/s:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={s/np.sqrt(len(x)):.6f}')
x=ics[5]
for n,m in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
 q=x[m];print(n,'dates',len(q),'IC',f'{q.mean():.6f}' if len(q) else None,'ICIR',f'{q.mean()/q.std(ddof=1):.6f}' if len(q)>1 else None,'hit',f'{(q>0).mean():.4f}' if len(q) else None)
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('signal_cell_coverage',f'{F.notna().mean().mean():.6f}','mean_daily_rank_turnover',f'{np.mean(turn):.6f}')
mx=-1;who='';cells=0
for n,s in L.items():
 z=pd.concat([F.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('library',n,'rho',f'{rho:.6f}','common_cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'active_library_records',len(glob.glob('factors/*.json')))
