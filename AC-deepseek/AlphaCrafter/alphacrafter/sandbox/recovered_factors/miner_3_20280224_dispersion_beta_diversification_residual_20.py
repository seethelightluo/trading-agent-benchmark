"""Miner_3 single-idea research: dispersion-beta diversification residual; data visible through 2028-02-23."""
import pandas as pd,numpy as np,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2028-02-23')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None); v=R.rolling(20,min_periods=15).std(); M=R.mean(axis=1)
# Daily cross-asset disagreement; higher signal = low sensitivity of asset shocks to broad dispersion.
D=R.std(axis=1); raw=pd.DataFrame({a:-R[a].abs().rolling(20,min_periods=15).corr(D) for a in A})
def res(x,*cs):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(k)) for k,z in enumerate(cs)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
F=res(raw,peer,v)
# Orthogonality comparators reproducible from active definitions; this is diagnostic only unless admission gates pass.
mom=(P/P.shift(20)-1)/(v+1e-12); v5=R.rolling(5,min_periods=4).std();L={'risk_adjusted_momentum':mom,'volnorm_reversal':-(P/P.shift(5)-1)/(v5+1e-12),'realized_volatility':v,'peer_crowding':peer}
Mneg=M.where(M<0);Mpos=M.where(M>0);L['downside_beta_asymmetry']=pd.DataFrame({a:-(R[a].where(M<0).rolling(30,min_periods=10).cov(Mneg)/Mneg.rolling(30,min_periods=10).var()-R[a].where(M>0).rolling(30,min_periods=10).cov(Mpos)/Mpos.rolling(30,min_periods=10).var()) for a in A})
print('FACTOR dispersion_beta_diversification_residual_20 visible_through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; x=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 s=pd.Series(dict(x),dtype=float);ics[h]=s; sd=s.std(ddof=1);print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(s)):.6f}')
for n,mask in [('2026',ics[10].index<'2027-01-01'),('2027', (ics[10].index>='2027-01-01')&(ics[10].index<'2028-01-01')),('2028',ics[10].index>='2028-01-01')]:
 s=ics[10][mask];print('regime',n,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else None,'hit',round((s>0).mean(),4))
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
mx=0
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library_diagnostic',n,'rho',f'{rho:.6f}','cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('max_abs_library_correlation_diagnostic',f'{mx:.6f}','against',who,'common_cells',cells,'admitted_json_records',len(glob.glob('factors/*.json')))
