"""One candidate: library-orthogonal continuous BTC volatility-interaction beta."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']; R=P.pct_change(); b=R.BTC
# A continuous (not thresholded) BTC-volatility interaction: sensitivity to BTC returns when BTC realized volatility is above/below its own trailing normal level.
v=b.rolling(10,min_periods=8).std(); vz=(v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std(); interaction=b*vz.shift(1)
raw=pd.DataFrame({a:R[a].rolling(60,min_periods=40).cov(interaction)/interaction.rolling(60,min_periods=40).var() for a in A})
base=cs(raw).shift(1)
def residual(d):
 y=base.loc[d]; X=pd.DataFrame({n:g.loc[d].reindex(A).fillna(0.) for n,g in S.items()},index=A);q=pd.concat([y.rename('y'),X],axis=1).dropna()
 if len(q)<8:return y*np.nan
 yy=q.y.to_numpy()-q.y.mean();xx=q.drop(columns='y').to_numpy();xx=(xx-xx.mean(0))/np.where(xx.std(0)>1e-12,xx.std(0),1)
 beta=np.linalg.solve(xx.T@xx+5*np.eye(xx.shape[1]),xx.T@yy);out=pd.Series(np.nan,index=A);out.loc[q.index]=yy-xx@beta;return out
cand=cs(pd.DataFrame({d:residual(d) for d in base.index}).T); fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];vs=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(ic):vs.append(ic);ns.append(len(q))
 if not vs:return {'dates':0}
 a=np.array(vs);return {'dates':len(a),'ic':round(float(a.mean()),6),'icir':round(float(a.mean()/a.std(ddof=1)),6),'hit':round(float((a>0).mean()),6),'mean_breadth':round(float(np.mean(ns)),3),'min_breadth':int(min(ns))}
print('FACTOR library_orthogonal_continuous_btc_volatility_interaction_beta_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in [1,5,10,20]:print('H',h,stat(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,p))
mx=-1;who='';e=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;e=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',e,'N_FACTORS',len(S))
