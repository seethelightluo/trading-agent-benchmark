"""Miner_2 single-idea test: residual breadth-confirmed intermediate continuation, visible through 2031-04-02."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-04-02')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# A 20-session residual trend relative to the contemporaneous cross-asset median return.
# Unlike isolated-move reversal, retain moves with broad residual-sign confirmation;
# the factor measures continuation of a shared but asset-specific residual direction.
e=R.sub(R.median(axis=1),axis=0)
m=e.rolling(20,min_periods=15).mean(); scale=e.rolling(40,min_periods=25).std().replace(0,np.nan)
sign=np.sign(m); breadth=sign.eq(sign.median(axis=1),axis=0).sum(axis=1)/sign.notna().sum(axis=1)
F=m.div(scale).mul(breadth,axis=0).loc[:END]
def met(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q))); widths.append(len(z))
 ic=pd.Series(dict(vals),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
M={}
for h in (1,5,10,20):
 ic,M[h]=met(h); print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic5,_=met(5)
for lab,mask in [('2020_2021',ic5.index.year<=2021),('2022_2023',ic5.index.year.isin([2022,2023])),('2024_2026',ic5.index.year.isin([2024,2025,2026])),('2027_2030',ic5.index.year.isin([2027,2028,2029,2030])),('2031_ytd',ic5.index.year==2031)]:
 x=ic5[mask]; print('REGIME_5D',lab,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): st.append(q)
evidence={}; mx=0.; most=None; complete=True
for p in glob.glob('factors/*.json'):
 try:j=json.load(open(p))
 except:continue
 if p.endswith('.bak') or j.get('validation',{}).get('status')=='DEPRECATED':continue
 fid=j.get('factor_id',os.path.basename(p)); key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 hits=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not hits: evidence[fid]={'rho':None,'valid_date_correlations':0,'common_signal_cells':0};complete=False;continue
 try:L=pd.read_pickle(max(hits,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 except Exception:L=pd.DataFrame()
 cs=[];cells=0
 for d in F.index.intersection(L.index):
  z=pd.concat([F.loc[d].rename('a'),L.loc[d].rename('b')],axis=1).dropna();cells+=len(z)
  if len(z)>=8:
   q=spearmanr(z.a,z.b).statistic
   if np.isfinite(q):cs.append(q)
 rho=float(np.mean(cs)) if cs else np.nan
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'valid_date_correlations':len(cs),'common_signal_cells':cells}
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(rho);most=fid
print('FACTOR residual_breadth_confirmed_intermediate_continuation_20v40obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'implied_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps(M,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'EVIDENCE_COMPLETE',complete,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_2_20310403_residual_breadth_confirmed_intermediate_continuation_20v40obs_signal.pkl')
