"""Miner 2 revalidation: Volume-Shock Short-Horizon Reversal, cutoff 2035-01-03."""
import os,glob,json,warnings,pickle
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-01-03'); FID='miner_2_volume_shock_short_reversal_5v20x60obs'
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
D={a:load(a) for a in A}; C=pd.DataFrame({a:D[a].close.astype(float) for a in A}).loc[:END]; V=pd.DataFrame({a:D[a].volume.astype(float) for a in A}).loc[:END]
F=-C.pct_change(5)*np.log(V.rolling(5,min_periods=4).mean()/V.rolling(60,min_periods=40).mean()).clip(-3,3)
def ev(h):
 y=C.shift(-h).div(C).sub(1); out=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): out.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd) if sd else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in(1,5,10,20): s,M[h]=ev(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
s,_=ev(1)
for name,yrs in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2034',[2032,2033,2034]),('2035',[2035])]:
 x=s[s.index.year.isin(yrs)];print('REGIME_1D',name,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
# exact novel/correlation evidence: compare against stored signal panels when present; all active factors must be evidenced
corr=[]; missing=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')!='EFFECTIVE' or j.get('factor_id')==FID:continue
  cand=glob.glob('scripts/*'+j['factor_id']+'*signal.pkl')
  if not cand: missing.append(j['factor_id']);continue
  x=pickle.load(open(cand[-1],'rb'))
  if isinstance(x,pd.DataFrame): x=x.reindex(index=F.index,columns=F.columns)
  vals=[]
  for dt in F.index:
   z=pd.concat([F.loc[dt],x.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
  if not vals: missing.append(j['factor_id'])
  else:corr.append((float(np.nanmax(vals)),j['factor_id'],len(vals)))
 except Exception as e: missing.append(p+':'+str(e))
print('STABILITY',float(np.nanmean(st)), 'TURNOVER',1-float(np.nanmean(st)),'COVERAGE',float(F.notna().mean().mean()),'ACTIVE',float(F.notna().sum(axis=1).mean()))
print('LIBRARY',sorted(corr,reverse=True)[:5],'MISSING',missing,'COUNT',len(corr))
pickle.dump(F,open('scripts/miner_2_20350104_volume_shock_short_reversal_revalidation_signal.pkl','wb'))
