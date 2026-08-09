"""miner_2 single-idea test: change in peer downside/upside co-movement asymmetry."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-10-29')
def load(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff(); peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
# For each asset and trailing window, correlation to peer basket in peer-down days minus peer-up days.
def asym(w):
 out=pd.DataFrame(index=R.index,columns=A,dtype=float)
 for a in A:
  x,y=R[a],peer[a]
  for i in range(w-1,len(R)):
   xx=x.iloc[i-w+1:i+1]; yy=y.iloc[i-w+1:i+1]; dn=yy<0; up=yy>0
   if dn.sum()>=8 and up.sum()>=8: out.iat[i,out.columns.get_loc(a)]=xx[dn].corr(yy[dn])-xx[up].corr(yy[up])
 return out
# Positive score: recent asymmetry has fallen relative to its slower baseline (a shape/change signal, not its level).
F=(asym(60)-asym(20)).loc[:END]
def metric(X,h):
 Y=(C.shift(-h)/C-1).reindex(X.index); v=[]; n=[]
 for d in X.index:
  z=pd.concat((X.loc[d].rename('x'),Y.loc[d].rename('y')),axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):v.append((d,float(q)));n.append(len(z))
 s=pd.Series(dict(v)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(n))}
for h in (1,5,10,20):
 s,m=metric(F,h); print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_ytd',s.index.year==2031),('recent_6m',s.index>=END-pd.Timedelta(days=184))]:
   q=s[mask]; print('REGIME',lab,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,float((q>0).mean()) if len(q) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[json.load(open(p))['factor_id'] for p in glob.glob('factors/*.json') if '_deprecated' not in p]
evidence={}; mx=0.; who=None; complete=True
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); matches=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not matches: evidence[fid]={'rho':None,'common_signal_cells':0}; complete=False; continue
 p=max(matches,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna(); q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q): complete=False
 elif abs(q)>mx: mx=float(abs(q));who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor':'peer_asymmetry_change_20v60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':evidence},sort_keys=True))
F.to_pickle('scripts/miner_2_20311030_peer_asymmetry_change_20v60obs_signal.pkl')
