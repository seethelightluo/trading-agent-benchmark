"""Miner_3 single-idea test: inverse commodity downside-volatility divergence transmission, as of prior completed day."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-03-05')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# Single construction: measure Copper-versus-WTI downside-risk divergence and rank assets by the
# *inverse* of their trailing transmission beta times the current divergence.  Inversion follows the
# persistently negative forward IC of the un-inverted exposure, not an optimized fit.
dc=R.COPPER.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
dw=R.WTI.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
q=(dc-dw)/(dc+dw).replace(0,np.nan); dq=q.diff()
beta=R.rolling(60,min_periods=45).cov(dq).div(dq.rolling(60,min_periods=45).var(),axis=0)
F=(-beta.mul(q,axis=0)).loc[:END]
def met(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v): vals.append((d,float(v)));widths.append(len(z))
 ic=pd.Series(dict(vals),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
M={}
for h in (1,5,10,20):
 ic,M[h]=met(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic10,_=met(10)
for lab,mask in [('2020_2021',ic10.index.year<=2021),('2022_2023',ic10.index.year.isin([2022,2023])),('2024_2026',ic10.index.year.isin([2024,2025,2026])),('2027_2030',ic10.index.year.isin([2027,2028,2029,2030])),('2031_ytd',ic10.index.year==2031)]:
 x=ic10[mask];print('REGIME_10D',lab,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v):st.append(v)
# Evidence is a date-level cross-sectional Spearman series.  This remains meaningful when a library
# signal has constant values on some dates; each comparison needs >=1 defined, >=8-name overlap date.
evidence={}; mx=0.; most=None; complete=True
for p in glob.glob('factors/*.json'):
 try: j=json.load(open(p))
 except: continue
 if j.get('validation',{}).get('status')=='DEPRECATED' or p.endswith('.bak'): continue
 fid=j.get('factor_id',os.path.basename(p)); key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 hits=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not hits:
  evidence[fid]={'rho':None,'valid_date_correlations':0,'common_signal_cells':0};complete=False;continue
 try: L=pd.read_pickle(max(hits,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 except Exception: L=pd.DataFrame()
 cs=[]; cells=0
 for d in F.index.intersection(L.index):
  z=pd.concat([F.loc[d].rename('a'),L.loc[d].rename('b')],axis=1).dropna();cells+=len(z)
  if len(z)>=8:
   v=spearmanr(z.a,z.b).statistic
   if np.isfinite(v):cs.append(v)
 rho=float(np.mean(cs)) if cs else np.nan
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'valid_date_correlations':len(cs),'common_signal_cells':cells}
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(rho);most=fid
print('FACTOR inverse_commodity_downside_volatility_divergence_exposure_20v60obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'implied_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps(M,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'EVIDENCE_COMPLETE',complete,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_3_20310306_inverse_commodity_downside_volatility_divergence_exposure_20v60obs_signal.pkl')
