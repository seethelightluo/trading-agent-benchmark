"""One candidate: directional run-length asymmetry (20 observations).
The score is (longest positive-return run minus longest negative-return run)/20.
It distinguishes persistence concentrated in advances from persistence concentrated in declines,
rather than measuring simple momentum or alternation frequency.
"""
import glob,json,os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-02-07')
def close(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=np.log(P/P.shift())
def runmax(x,sgn):
 q=(np.asarray(x)*sgn>0); best=cur=0
 for v in q:
  cur=cur+1 if v else 0;best=max(best,cur)
 return best
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 F[a]=R[a].rolling(20,min_periods=20).apply(lambda x:(runmax(x,1)-runmax(x,-1))/20,raw=True)
def metric(h):
 fw=P.shift(-h)/P-1; rows=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: rows.append((d,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(rows)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20]:
 s,m=metric(h);print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for n,mask in [('2020_2022',s.index.year<=2022),('2023_2024',s.index.year.isin([2023,2024])),('2025_2026',s.index.year.isin([2025,2026])),('2027_2028',s.index.year.isin([2027,2028])),('2029_ytd',s.index.year==2029)]:
   q=s[mask];print('REGIME_10D',n,json.dumps({'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit':float((q>0).mean()) if len(q) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
F.to_pickle('scripts/miner_2_20290208_directional_run_length_asymmetry_20obs_signal.pkl')
# Resolve each admitted-factor artifact by factor-id search, with explicit legacy aliases.
alias={'miner_2_realized_volatility_20obs':'miner_2_20260716_realized_volatility_20obs_signal.pkl','miner_2_volume_confirmed_drawdown_recovery_60d':'miner_2_20261105_volume_confirmed_drawdown_recovery_60d_signal.pkl','inverse_return_serial_dependence_20obs':'miner_2_20270701_inverse_return_serial_dependence_20obs_signal.pkl','downside_concentration_continuation_10v40obs':'miner_2_20271118_downside_concentration_continuation_10v40obs_signal.pkl','upside_concentration_exhaustion_10v40obs':'miner_2_20271202_upside_concentration_exhaustion_10v40obs_signal.pkl','miner_2_standardized_jump_asymmetry_20v40obs':'miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl','miner_2_range_compressed_intermediate_continuation_10to20x10v40obs':'miner_2_20280504_range_compressed_intermediate_continuation_10to20x10v40obs_signal.pkl','miner_2_normalized_overnight_gap_reversal_5v20obs':'miner_2_20280824_normalized_overnight_gap_reversal_5v20obs_signal.pkl','miner_1_volnorm_reversal_5obs':'miner_1_20260716_volnorm_reversal5_signal.pkl'}
rows=[];missing=[]; effective=[]
for path in glob.glob('factors/*.json'):
 d=json.load(open(path))
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 fid=d['factor_id'];effective.append(fid); fn=alias.get(fid); hits=glob.glob('scripts/*'+fid+'*signal.pkl') if not fn else ['scripts/'+fn]
 if not hits or not os.path.exists(sorted(hits)[-1]):missing.append(fid);continue
 G=pd.read_pickle(sorted(hits)[-1]);x,y=F.align(G,join='inner',axis=0);z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna()
 rho=float(spearmanr(z.x,z.y).statistic) if len(z)>2 else np.nan;rows.append((fid,len(z),rho));print('LIBRARY_CORR',fid,len(z),rho)
print('LIBRARY_MISSING',json.dumps(missing))
if not missing and len(rows)==len(effective) and all(np.isfinite(q[2]) for q in rows):
 b=max(rows,key=lambda q:abs(q[2]));print('LIBRARY_MAX',json.dumps({'factor':b[0],'cells':b[1],'rho':b[2],'max_abs_library_correlation':abs(b[2]),'audited_factors':len(rows)}))
