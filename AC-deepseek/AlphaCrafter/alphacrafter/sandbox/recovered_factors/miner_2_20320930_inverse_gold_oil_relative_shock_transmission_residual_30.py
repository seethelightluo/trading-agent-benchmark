"""Single candidate: inverse gold-oil relative-shock transmission residual, 30 sessions."""
import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-09-29')
def read(a,col='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=END,col],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(1)
def resid(x,*c):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(j)) for j,z in enumerate(c)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
v=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
def beta(z,mask,w=30,n=10):
 return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(z.where(mask))/z.where(mask).rolling(w,min_periods=n).var() for a in A})
dba=-(beta(M,M<0)-beta(M,M>0))
# Relative gold-versus-oil return shock captures a safe-haven/inflation-risk split.
# Score is inverse difference in asset exposure in high versus ordinary magnitude shocks,
# residualized against common cross-asset risk characteristics.
z=R.XAU-R.WTI; zs=(z-z.rolling(60,min_periods=40).mean())/(z.rolling(60,min_periods=40).std()+1e-12); high=zs.abs()>=zs.abs().rolling(60,min_periods=40).median()
F=resid(-(beta(zs,high)-beta(zs,~high)),v,peer,dba,trend)
def evaluate(f,h):
 ics=[]; ns=[]
 for t in P.index:
  q=pd.concat([f.loc[t],(P.shift(-h)/P-1).loc[t]],axis=1).dropna()
  if len(q)>=8: ics.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()));ns.append(len(q))
 x=np.array(ics); return dict(ic=float(x.mean()),icir=float(x.mean()/(x.std(ddof=1)+1e-12)),hit=float((x>0).mean()),dates=len(x),mean_n=float(np.mean(ns)))
# broad proxy-library novelty screen, including closest related macro transmission families
D={}
for nm,series in [('dxy',read('DXY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)),('vix',read('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)),('usdjpy',read('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)),('usdcny',read('USDCNY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)),('oil',R.WTI),('copper',R.COPPER)]:
 D[nm]=resid(-(beta(series,series>0)-beta(series,series<0)),v,peer,dba,trend)
D['rate_spread']=resid(-(beta(R.US10Y-R.CN10Y,(R.US10Y-R.CN10Y).abs()>=(R.US10Y-R.CN10Y).abs().rolling(60,min_periods=40).median())-beta(R.US10Y-R.CN10Y,(R.US10Y-R.CN10Y).abs()<(R.US10Y-R.CN10Y).abs().rolling(60,min_periods=40).median())),v,peer,dba,trend)
D['trend']=trend/(v+1e-12); D['vol']=v; D['peer']=peer; D['dba']=dba
cors=[]
for nm,x in D.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna()
 cors.append((abs(q.f.rank().corr(q.x.rank())),nm,len(q)))
print('FACTOR inverse_gold_oil_relative_shock_transmission_residual_30'); print('endpoint',P.index.max().date(),'assets',len(A),'factor cells',int(F.notna().sum().sum()),'coverage',round(F.notna().mean().mean(),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6))
for h in [1,5,10,20]: print('H',h,evaluate(F,h))
print('REGIME_2026_2027',evaluate(F.loc[(F.index>='2026-01-01')&(F.index<='2027-12-31')],20));print('REGIME_2028_PLUS',evaluate(F.loc[F.index>='2028-01-01'],20));print('MAX_LIBRARY_PROXY_CORR',max(cors), 'top5',sorted(cors,reverse=True)[:5])
PY
