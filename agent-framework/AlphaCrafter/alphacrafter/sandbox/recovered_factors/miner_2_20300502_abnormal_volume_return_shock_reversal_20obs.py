"""Candidate: abnormal-volume return-shock reversal, visible only through 2030-05-01."""
import glob,json,os
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-05-01')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d[['close','volume']].astype(float).where(lambda x:x>0)
D={a:load(a) for a in A}; P=pd.DataFrame({a:D[a].close for a in A}); V=pd.DataFrame({a:D[a].volume for a in A}); R=P.pct_change()
# High score: unusually high volume accompanying an unusually negative one-day return.
# Both components are scaled only with trailing 20-observation statistics.
vz=(np.log(V).sub(np.log(V).rolling(20,min_periods=15).mean())).div(np.log(V).rolling(20,min_periods=15).std())
rz=R.div(R.rolling(20,min_periods=15).std())
F=(-vz*rz).replace([np.inf,-np.inf],np.nan)
print('FACTOR abnormal_volume_return_shock_reversal_20obs visible_through',END.date(),'panel_dates',len(F),'instruments',len(A))
print('COVERAGE',{'signal_cell_coverage':float(F.notna().mean().mean()),'mean_valid_instruments':float(F.notna().sum(1).mean())})
def summ(s,b=[]): return {'ic_dates':len(s),'daily_paper_ic':float(s.mean()) if len(s) else None,'daily_paper_icir':float(s.mean()/s.std(ddof=1)) if len(s)>1 else None,'ic_hit_ratio':float((s>0).mean()) if len(s) else None,'mean_valid_instruments':float(np.mean(b)) if b else None}
def test(h):
 out=[];b=[]; fw=P.shift(-h)/P-1
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:out.append((dt,spearmanr(z.f,z.r).statistic));b.append(len(z))
 s=pd.Series(dict(out));print('HORIZON',h,summ(s,b))
 for n,l,u in [('2025_2026','2025-01-01','2026-12-31'),('2027_2028','2027-01-01','2028-12-31'),('2029_2030','2029-01-01','2030-05-01')]:print('REGIME',h,n,summ(s.loc[l:u]))
 return s
S={h:test(h) for h in [1,5,10,20]}
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('TURNOVER',{'rank_stability_1d':float(np.mean(st)),'mean_rank_turnover_1d':float((1-np.mean(st))/2),'rank_dates':len(st)})
# Required full library audit using persisted signal artifacts.
mx=-1; who=None; cells=0; loaded=0; missing=[]
for fp in glob.glob('factors/*.json'):
 try:
  meta=json.load(open(fp));
  if meta.get('validation',{}).get('status')!='EFFECTIVE':continue
  path=meta.get('signal_artifact');
  if not path or not os.path.exists(path):missing.append(meta.get('factor_id'));continue
  X=pd.read_pickle(path)
  if isinstance(X,pd.Series): X=X.unstack() if isinstance(X.index,pd.MultiIndex) else X.to_frame()
  X.index=pd.to_datetime(X.index); X=X.reindex(index=F.index,columns=A)
  z=pd.DataFrame({'x':F.stack(),'y':X.stack()}).dropna()
  if len(z)<8: missing.append(meta.get('factor_id')+' insufficient');continue
  rho=spearmanr(z.x,z.y).statistic;loaded+=1;cells+=len(z)
  if abs(rho)>mx:mx=abs(rho);who=meta.get('factor_id')
 except Exception as e: missing.append(os.path.basename(fp)+':'+str(e))
print('LIBRARY_AUDIT',{'effective_factors_loaded':loaded,'missing':missing,'max_abs_library_correlation':mx,'max_factor':who,'common_valid_cells_sum':cells})
F.to_pickle('scripts/miner_2_20300502_abnormal_volume_return_shock_reversal_20obs_signal.pkl')
print('DECISION', 'EFFECTIVE' if abs(S[5].mean())>=.007 and abs(S[5].mean()/S[5].std(ddof=1))>=.084 and mx<.5 and not missing else 'REJECT')
