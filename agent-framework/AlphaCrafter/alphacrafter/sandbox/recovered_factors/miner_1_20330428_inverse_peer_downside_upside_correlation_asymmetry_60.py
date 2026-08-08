"""One idea: inverse peer downside-versus-upside correlation asymmetry (60 sessions).
For each asset, estimate its average correlation with peers separately on dates
when the equal-weight cross-asset return is negative versus positive. Assets
whose downside co-movement exceeds upside co-movement are relatively fragile;
the inverse asymmetry should predict medium-horizon relative resilience.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']; r=P.pct_change(); mkt=r.median(axis=1)
raw=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in range(60,len(r)+1):
    rr=r.iloc[t-60:t]; up=rr.loc[mkt.loc[rr.index]>0]; down=rr.loc[mkt.loc[rr.index]<0]
    if len(up)>=12 and len(down)>=12:
        cu=up.corr(); cd=down.corr()
        raw.iloc[t-1]=[cd.loc[a].drop(a).mean()-cu.loc[a].drop(a).mean() for a in A]
# fully trailing calculation; lag to ensure completed-session availability
cand=(-raw.sub(raw.median(axis=1),axis=0)).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(ic):v.append(ic);b.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR inverse_peer_downside_upside_correlation_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME20',n,st(20,p))
mx=-1;who=''; evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
