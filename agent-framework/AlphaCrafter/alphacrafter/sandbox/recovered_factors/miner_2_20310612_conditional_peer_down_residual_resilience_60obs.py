"""miner_2: conditional peer-down residual resilience, 60 sessions."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-06-11')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff(); P=pd.DataFrame({a:R.drop(columns=a).median(axis=1) for a in A}); residual=R-P
# On broad peer-negative sessions, measure each asset's average peer-relative return, standardized by conditional residual volatility.
neg=P<0; cnt=neg.rolling(60,min_periods=30).sum(); mu=residual.where(neg).rolling(60,min_periods=30).mean(); sd=residual.where(neg).rolling(60,min_periods=30).std(); F=(mu/sd*np.sqrt(cnt)).loc[:END]
def met(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[]; ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,q));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in (1,5,10,20):
 s,m=met(F,h); print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_YTD',s.index.year==2031)]:
   q=s[mask];print('REGIME',lab,len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)),float((q>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except: pass
files=glob.glob('scripts/*_signal.pkl'); mx=0;who=None;complete=True; ev={}
for fid in active:
 # exact normalized id match first, then most recently written matching artifact
 key=fid.split('_',2)[-1]; cand=[p for p in files if key in os.path.basename(p)]
 if not cand: complete=False;ev[fid]={'rho':None,'common_signal_cells':0};continue
 p=max(cand,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna(); q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q): complete=False
 elif abs(q)>mx: mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor':'conditional_peer_down_residual_resilience_60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_2_20310612_conditional_peer_down_residual_resilience_60obs_signal.pkl')
