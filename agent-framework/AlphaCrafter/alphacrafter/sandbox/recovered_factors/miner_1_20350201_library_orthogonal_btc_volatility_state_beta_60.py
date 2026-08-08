"""One candidate: library-orthogonal BTC volatility-state beta differential."""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']; r=P.pct_change()
# Difference between each asset's BTC beta during high BTC realized-volatility states
# and its unconditional beta. State is known at the close and signal is lagged.
br=r['BTC']; bv=br.rolling(10,min_periods=8).std(); state=bv>bv.rolling(60,min_periods=40).quantile(.70)
def conditional_beta(x,y,e,w=60):
    q=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1).where(lambda d:d.e)
    return q.x.rolling(w,min_periods=12).cov(q.y)/q.y.rolling(w,min_periods=12).var()
raw=pd.DataFrame({a:conditional_beta(r[a],br,state)-r[a].rolling(60,min_periods=40).cov(br)/br.rolling(60,min_periods=40).var() for a in A})
base=cs(raw).shift(1)
# Ridge residualization gives a maintained, explicitly library-distinct signal.
def resid(dt):
    y=base.loc[dt]; X=pd.DataFrame({n:g.loc[dt].reindex(A).fillna(0.) for n,g in S.items()},index=A)
    q=pd.concat([y.rename('y'),X],axis=1).dropna()
    if len(q)<8:return y*np.nan
    yy=q.y.to_numpy(); xx=q.drop(columns='y').to_numpy(); xx=(xx-xx.mean(0))/np.where(xx.std(0)>1e-12,xx.std(0),1); yy=yy-yy.mean()
    b=np.linalg.solve(xx.T@xx+5*np.eye(xx.shape[1]),xx.T@yy)
    out=pd.Series(np.nan,index=A);out.loc[q.index]=yy-xx@b;return out
cand=cs(pd.DataFrame({d:resid(d) for d in base.index}).T)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]; vals=[];breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR library_orthogonal_btc_volatility_state_beta_differential_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,stats(10,p))
mx=-1;who='';evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
