"""Revalidate miner_3 conditional USDCNY impulse exposure through last completed date."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-11-13')
def close(p):
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
fx=close('../persistent/index_data/USDCNY.csv').reindex(C.index).ffill()
r=np.log(C).diff(); fr=np.log(fx).diff()
beta=r.rolling(50,min_periods=35).cov(fr).div(fr.rolling(50,min_periods=35).var(),axis=0)
F=beta.mul(-fr.rolling(10,min_periods=10).sum(),axis=0).loc[:END]
def calc(h):
 future=(C.shift(-h)/C-1).reindex(F.index); out=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),future.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): out.append((d,float(q))); widths.append(len(z))
 ic=pd.Series(dict(out),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
allm={}
for h in (1,5,10,20):
 _,allm[h]=calc(h); print('HORIZON',h,json.dumps(allm[h],sort_keys=True))
ic,_=calc(20)
for lab,mask in [('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask]; print('REGIME',lab,len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[json.load(open(p))['factor_id'] for p in glob.glob('factors/*.json') if '_deprecated' not in p]
files=glob.glob('scripts/*_signal.pkl'); evidence={}; mx=0.0
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 choices=[p for p in files if key in os.path.basename(p)]
 if not choices: evidence[fid]=None; mx=np.inf; print('CORR',fid,'MISSING'); continue
 p=max(choices,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('a'),lib.stack().rename('b')],axis=1).dropna(); rho=float(spearmanr(z.a,z.b).statistic) if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame(); rho=np.nan
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'cells':len(z),'file':p}; mx=max(mx,abs(rho)) if np.isfinite(rho) else np.inf
 print('CORR',fid,len(z),rho)
print('PANEL',F.index.min().date(),END.date(),len(F),float(F.notna().mean().mean()),float(F.notna().sum(axis=1).mean()),float(np.mean(st)),float(1-np.mean(st)))
print('MAXCORR',mx,'COMPLETE',all(v is not None and v['rho'] is not None for v in evidence.values()),'N',len(active))
F.to_pickle('scripts/miner_3_20301114_conditional_usdcny_impulse_exposure_10v50obs_revalidation_signal.pkl')
