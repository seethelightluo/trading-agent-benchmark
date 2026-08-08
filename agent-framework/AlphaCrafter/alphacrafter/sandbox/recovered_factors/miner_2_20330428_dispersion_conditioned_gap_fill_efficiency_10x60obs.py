"""Miner_2 candidate: dispersion-conditioned gap-fill efficiency (10 sessions)."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-04-27'); FID='miner_2_dispersion_conditioned_gap_fill_efficiency_10x60obs'
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[['open','high','low','close','volume']].astype(float)
D={a:load(a) for a in A}; C=pd.DataFrame({a:D[a]['close'] for a in A}).loc[:END]
R=C.pct_change(); disp=R.std(axis=1); stress=disp/disp.rolling(60,min_periods=40).median()
F=pd.DataFrame(index=C.index,columns=A,dtype=float)
# Positive: a gap is efficiently retraced during a cross-asset dispersion shock.
# Normalize both legs by ex-ante own volatility; gate continuously by market dispersion.
for a,d in D.items():
 d=d.reindex(C.index); vol=R[a].rolling(20,min_periods=15).std().replace(0,np.nan)
 gap=d.open.div(d.close.shift(1)).sub(1).div(vol)
 intra=d.close.div(d.open).sub(1).div(vol)
 fill=(-gap*intra).clip(-12,12)
 F[a]=(fill*stress.clip(.5,2.5)).rolling(10,min_periods=8).mean()
def ev(h):
 y=C.shift(-h).div(C).sub(1);rows=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):rows.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(rows),dtype=float);sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd) if sd else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
MET={}
for h in (1,5,10,20):
 s,MET[h]=ev(h);print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
s,_=ev(5)
for n,ys in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('recent_2032_2033',[2032,2033])]:
 x=s[s.index.year.isin(ys)];sd=x.std(ddof=1);print('REGIME_5D',n,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/sd) if len(x)>1 and sd else None,'hit',float((x>0).mean()) if len(x) else None)
stab=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):stab.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
evidence={};complete=True;mx=0.;most=None
for fid in active:
 key=fid
 for pre in ('miner_1_','miner_2_','miner_3_'):key=key.replace(pre,'')
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths:evidence[fid]={'rho':None,'common_signal_cells':0,'file':None};complete=False;continue
 p=max(paths,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except:z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(stab)),'implied_rank_turnover',float(1-np.mean(stab)))
print('DECAY',json.dumps({str(k):v for k,v in MET.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_2_20330428_dispersion_conditioned_gap_fill_efficiency_10x60obs_signal.pkl')
