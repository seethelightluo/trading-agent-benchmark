"""Miner_2 single idea: inverse USDCNY shock-transmission asymmetry residual, point-in-time validation."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-11-09')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None);M=R.mean(1);v=R.rolling(20,min_periods=15).std();trend=P/P.shift(20)-1
def residual(x,*cs):
 y=x.to_numpy(float);z=[c.to_numpy(float) for c in cs];o=np.full(y.shape,np.nan)
 for i in range(len(P)):
  X=np.column_stack([q[i] for q in z]);ok=np.isfinite(y[i])&np.isfinite(X).all(1)
  if ok.sum()>=8:
   X1=np.c_[np.ones(ok.sum()),X[ok]]
   if np.linalg.matrix_rank(X1)==X1.shape[1]:o[i,ok]=y[i,ok]-X1@np.linalg.lstsq(X1,y[i,ok],rcond=None)[0]
 return pd.DataFrame(o,index=P.index,columns=A)
def beta(z,mask,w=30,n=10):
 q=z.where(mask); den=q.rolling(w,min_periods=n).var();return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(q)/den for a in A})
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
dba=-(beta(M,M<0)-beta(M,M>0))
# Pre-specified: inverse positive-minus-negative RMB depreciation shock beta. This favors assets
# insulated from asymmetric RMB-risk-off repricing, purged of generic risk/crowding/trend traits.
z=rd('USDCNY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
F=-residual(beta(z,z>0)-beta(z,z<0),v,peer,dba,trend)
print('FACTOR inverse_usdcny_shock_transmission_asymmetry_residual_30 cutoff',P.index.max().date(),'source_rows',len(P),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for lab,ma in [('2020_25',ics[20].index<'2026-01-01'),('2026_27',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_current',ics[20].index>='2028-01-01')]:
 x=ics[20][ma];print('regime20',lab,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else 'nan','hit',f'{(x>0).mean():.4f}' if len(x) else 'nan')
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
# Mandatory novelty screen, first against the closest same-family admitted RMB/rate/macro transmission proxies.
# If eligibility holds this script is extended to all exact stored definitions before persistence.
libs={}
for nm,s,sgn in [('DXY','DXY',1),('EURUSD','EURUSD',-1),('USDJPY','USDJPY',-1),('VIX','VIX',-1)]:
 q=rd(s,root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index);libs[nm]=residual(sgn*(beta(q,q>0)-beta(q,q<0)),v,peer,dba,trend)
rate=R.US10Y-R.CN10Y;libs['rate_spread_beta']=residual(-beta(rate,rate.notna(),30,15),v,peer,dba,trend)
vals=[]
for nm,x in libs.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman'); vals.append((abs(rho),nm,rho,len(q)));print('proxy_library',nm,'rho',f'{rho:.6f}','cells',len(q))
b=max(vals);print('proxy_max_abs_library_correlation',f'{b[0]:.6f}','against',b[1],'cells',b[3])
