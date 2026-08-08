"""miner_2: downside-versus-upside peer-correlation asymmetry (one factor idea)."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-11-13')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); r=np.log(C).diff(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
# For each trailing 60-session window, measure correlation conditional on peer-market down days minus up days.
def asym(a):
 x=r[a]; p=peer[a]; out=[]
 for j in range(len(r)):
  z=pd.concat([x.iloc[max(0,j-59):j+1],p.iloc[max(0,j-59):j+1]],axis=1).dropna(); z.columns=['x','p']
  dn=z[z.p<0]; up=z[z.p>=0]
  out.append(spearmanr(dn.x,dn.p).statistic-spearmanr(up.x,up.p).statistic if len(dn)>=12 and len(up)>=12 else np.nan)
 return pd.Series(out,index=r.index)
F=pd.DataFrame({a:asym(a) for a in A}).loc[:END]
def metrics(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); vals=[];ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for direction,X in [('asymmetry',F),('inverse_asymmetry',-F)]:
 for h in (1,5,10,20):
  s,m=metrics(X,h); print('HORIZON',direction,h,json.dumps(m,sort_keys=True))
  if h==10:
   for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year>=2027)]:
    q=s[mask];print('REGIME',direction,lab,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,float((q>0).mean()) if len(q) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[json.load(open(p))['factor_id'] for p in glob.glob('factors/*.json') if '_deprecated' not in p]; files=glob.glob('scripts/*_signal.pkl');mx=0;who=None;complete=True
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); ms=[p for p in files if key in os.path.basename(p)]
 if not ms: complete=False;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(ms,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=float(spearmanr(z.x,z.l).statistic) if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 print('LIBRARY_CORR',fid,len(z),q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
print('SUMMARY',json.dumps({'factor':'downside_upside_peer_correlation_asymmetry_60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'observed_max':mx,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20301114_downside_upside_peer_correlation_asymmetry_60obs_signal.pkl')
