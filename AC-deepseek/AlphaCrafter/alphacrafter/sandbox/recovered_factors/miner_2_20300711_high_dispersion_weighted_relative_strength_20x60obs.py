"""miner_2 20300711: high-dispersion-weighted relative strength.
Tests whether cross-asset 20-day relative strength is more predictive when the
cross-sectional dispersion of those returns is high versus its trailing history.
Unlike prior reversal construction, the factor never changes the signal sign.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-07-10')
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index()
r20=np.log(C/C.shift(20)); disp=r20.std(axis=1,ddof=1)
# strictly contemporaneous factor inputs available at each end-of-day cursor
z=(disp-disp.rolling(60,min_periods=40).mean())/disp.rolling(60,min_periods=40).std()
# logistic [0,1], no reversal: emphasis rises continuously with dispersion
mult=1/(1+np.exp(-z.clip(-8,8)))
F=r20.sub(r20.median(axis=1),axis=0).mul(mult,axis=0).loc[:END]
def calc(h):
    future=(C.shift(-h)/C-1).reindex(F.index); vals=[]; nn=[]
    for d in F.index:
        q=pd.concat([F.loc[d].rename('f'),future.loc[d].rename('r')],axis=1).dropna()
        if len(q)>=8:
            v=spearmanr(q.f,q.r).statistic
            if np.isfinite(v): vals.append((d,float(v)));nn.append(len(q))
    ic=pd.Series(dict(vals)); sd=ic.std(ddof=1)
    return ic, {'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(nn))}
ALL={}
for h in (1,5,10,20):
    ic,ALL[h]=calc(h);print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=calc(5)
for lab,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask]; print('REGIME_5D',lab,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 q=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(q)>=8: st.append(float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
active=[]
for p in glob.glob('factors/*.json'):
 if p.endswith('.bak') or '_deprecated' in p: continue
 try:
  d=json.load(open(p))
  if d.get('validation',{}).get('status')=='EFFECTIVE': active.append(d['factor_id'])
 except Exception: pass
mx=0.; who=None; evidence={}; files=glob.glob('scripts/*_signal.pkl')
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 found=[p for p in files if key in os.path.basename(p)]
 if not found: mx=np.inf; evidence[fid]={'rho':None,'common_signal_cells':0}; print('LIBRARY_CORR',fid,'MISSING'); continue
 p=max(found,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); q=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna(); rho=float(spearmanr(q.f,q.l).statistic) if len(q)>=8 else np.nan
 except Exception: q=pd.DataFrame();rho=np.nan
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(q),'file':p}
 if not np.isfinite(rho): mx=np.inf
 elif abs(rho)>mx: mx=abs(rho);who=fid
 print('LIBRARY_CORR',fid,'cells',len(q),'spearman',rho)
print('FACTOR high_dispersion_weighted_relative_strength_20x60obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who,'AUDITED',len(active),'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_2_20300711_high_dispersion_weighted_relative_strength_20x60obs_signal.pkl')
