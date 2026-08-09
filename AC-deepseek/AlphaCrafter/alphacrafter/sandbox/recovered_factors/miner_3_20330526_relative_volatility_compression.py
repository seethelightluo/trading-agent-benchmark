"""Miner_3: relative realized-volatility compression factor, visible through 2033-05-25."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-05-25'); FID='miner_3_relative_volatility_compression_20obs'
def load(p):
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]
r=np.log(C).diff()
# Negative idiosyncratic relative realized volatility: lower-than-peer 20-observation risk is favored.
ov=r.rolling(20,min_periods=15).std()
F=-ov.div(ov.median(axis=1),axis=0)
def measure(h):
 y=C.shift(-h).div(C).sub(1); vals=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q)));widths.append(len(z))
 ic=pd.Series(dict(vals),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
M={}
for h in (1,5,10,20):
 _,M[h]=measure(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=measure(10)
for label,yr in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_2033',[2031,2032,2033])]:
 x=ic[ic.index.year.isin(yr)]; print('REGIME_10D',label,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(float(q))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except: pass
E={};complete=True;mx=0.;most=None
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 if fid=='miner_1_state_gated_volatility_expansion_10v60obs':key='state_gated_inverse_volatility_expansion_10v60obs'
 ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 try:
  L=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_3_20330526_relative_volatility_compression_signal.pkl')
