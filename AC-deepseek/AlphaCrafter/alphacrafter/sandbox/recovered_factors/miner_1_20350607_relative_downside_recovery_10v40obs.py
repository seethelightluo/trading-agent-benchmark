"""Miner 1: relative downside recovery strength; single interpretable candidate, cutoff 2035-06-06."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-06-06'); eps=1e-7
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).loc[:END]; r=np.log(C).diff()
# Recovery over 10 sessions, scaled by depth below the preceding 40-session peak.
# A positive score identifies assets recovering most efficiently from a meaningful prior drawdown.
peak=C.shift(10).rolling(40,min_periods=30).max(); depth=(1-C.shift(10)/peak).clip(lower=0)
F=(C/C.shift(10)-1)/(depth+0.02)
def metrics(h):
 y=C.shift(-h).div(C).sub(1); out=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20,40]:
 x,m=metrics(h);print('HORIZON',h,json.dumps(m,sort_keys=True))
x,_=metrics(10)
for label,ys in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_2033',[2031,2032,2033]),('2034_2035',[2034,2035])]:
 s=x[x.index.year.isin(ys)];print('REGIME_10D',label,'dates',len(s),'IC',float(s.mean()),'ICIR',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
mx=0.;most=None;complete=True;evidence={}
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','');key={'state_gated_volatility_expansion_10v60obs':'state_gated_inverse_volatility_expansion_10v60obs'}.get(key,key)
 pp=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not pp: evidence[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 try:
  L=pd.read_pickle(max(pp,key=os.path.getmtime)).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna();q=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL',F.index.min().date(),END.date(),len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active));print('EVIDENCE',json.dumps(evidence,sort_keys=True));F.to_pickle('scripts/miner_1_20350607_relative_downside_recovery_10v40obs_signal.pkl')
