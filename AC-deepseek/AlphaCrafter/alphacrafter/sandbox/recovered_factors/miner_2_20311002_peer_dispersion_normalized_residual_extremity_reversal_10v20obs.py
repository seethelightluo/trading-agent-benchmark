"""miner_2: peer-dispersion-normalized residual extremity reversal (single idea)."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-10-01')
def load(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# Leave-one-out peer benchmark prevents an asset influencing its own residual.
P=pd.DataFrame({a:R.drop(columns=a).median(axis=1) for a in A}); E=R-P
# Dense contrarian residual signal. Divide 10d abnormal return by contemporaneous
# 20d cross-asset peer dispersion: extremes in coherent/low-dispersion markets
# are hypothesized to mean-revert more strongly, without a sparse state gate.
peer_disp=pd.DataFrame({a:R.drop(columns=a).std(axis=1) for a in A})
denom=peer_disp.rolling(20,min_periods=12).mean().clip(lower=1e-7)
F=-(E.rolling(10,min_periods=7).sum()/denom).loc[:END]
def metric(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); vals=[]; nums=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append((d,q)); nums.append(len(z))
 s=pd.Series(dict(vals),dtype=float); s.index=pd.to_datetime(s.index); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(nums))}
for h in [1,5,10,20]:
 s,m=metric(F,h); print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for label,yr in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_YTD',[2031])]:
   q=s[s.index.year.isin(yr)]; print('REGIME',label,json.dumps({'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean())}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
mx=0.; who=None; complete=True
for fid in active:
 key=fid.split('_',2)[-1]; paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths: complete=False; print('LIBRARY_CORR',fid,'MISSING'); continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna(); q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame(); q=np.nan
 print('LIBRARY_CORR',fid,len(z),q)
 if not np.isfinite(q): complete=False
 elif abs(q)>mx: mx=abs(q);who=fid
print('SUMMARY',json.dumps({'factor':'peer_dispersion_normalized_residual_extremity_reversal_10v20obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20311002_peer_dispersion_normalized_residual_extremity_reversal_10v20obs_signal.pkl')
