"""miner_2 single idea: conditional cross-asset recovery after broad stress."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2032-03-31')
def load(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:load(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(1); v=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
def resid(x,*controls):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(controls)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
def beta(z,mask,w=30,n=10):
 return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(z.where(mask))/z.where(mask).rolling(w,min_periods=n).var() for a in A})
dba=-(beta(M,M<0)-beta(M,M>0)); loss=(R<0).rolling(20,min_periods=15).mean()
# Candidate: volatility-scaled return conditional on an already completed broad-market
# stress day. The score is a 20-session weighted mean of own next-session recovery,
# with weights equal to prior broad stress severity, then residualized from generic risk.
stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,4)
raw=R.mul(stress,axis=0).rolling(20,min_periods=12).sum().div(stress.rolling(20,min_periods=12).sum().replace(0,np.nan),axis=0).div(v+1e-12)
F=resid(raw,v,peer,dba,trend,loss)
print('FACTOR stress_day_recovery_efficiency_residual_20 endpoint',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 vals=[]; nums=[]; fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append((t,q.f.corr(q.r,method='spearman')));nums.append(len(q))
 x=pd.Series(dict(vals));ics[h]=x; print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(nums):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for n,mask in [('2026_27',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_current',ics[20].index>='2028-01-01')]:
 x=ics[20][mask];print('regime20',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
