import os, glob, json, pickle, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

# One idea revalidated: high-VIX, 20-session range-location reversal. This version
# also performs a strict finite-observation audit against the effective library.
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
def panel(field):
    return pd.concat([pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()[field].rename(a) for a in assets],axis=1)
C,H,L=panel('close'),panel('high'),panel('low')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close']
idx=C.index.intersection(v.index); cut=pd.Timestamp('2035-11-07')
C,H,L,v=C.loc[idx].loc[:cut],H.loc[idx].loc[:cut],L.loc[idx].loc[:cut],v.loc[idx].loc[:cut]
low20=L.rolling(20,min_periods=20).min(); width=H.rolling(20,min_periods=20).max()-low20
stress=v>v.rolling(60,min_periods=60).median()
sig=-(C-low20).div(width.replace(0,np.nan)).sub(.5).where(stress,0.)
pickle.dump(sig,open('scripts/miner_1_20351108_vix_stress_range_location_reversal_20v60obs_signal.pkl','wb'))
def ic_metrics(h):
    f=C.shift(-h).div(C).sub(1); out=[]; nn=[]
    for d in sig.index:
        z=pd.concat([sig.loc[d],f.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(r): out.append((d,r));nn.append(len(z))
    s=pd.Series(dict(out)); return s,{'ic':s.mean(),'icir':s.mean()/s.std(ddof=1),'hit':(s>0).mean(),'dates':len(s),'mean_n':np.mean(nn)}
print('FACTOR vix_stress_range_location_reversal_20v60obs; panel',C.index.min().date(),C.index.max().date(),'assets',len(assets))
print('stress_rate',round(stress.mean(),5),'coverage',round(sig.notna().mean().mean(),5),'active_dates',int(stress.sum()))
for h in [1,5,10,20,40]:
    s,m=ic_metrics(h); print('H',h,{k:round(v,5) for k,v in m.items()})
    if h==40:
      for n,a,b in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_current','2034-01-01','2035-11-07')]:
       q=s.loc[a:b]; print('REGIME',n,'dates',len(q),'ic',round(q.mean(),5),'icir',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),5))
print('rank_stability',round(sig.corrwith(sig.shift(),axis=1,method='spearman').mean(),5))
print('turnover_proxy',round((sig.rank(axis=1,pct=True)-sig.shift().rank(axis=1,pct=True)).abs().mean(axis=1).mean(),5))
# Exact effective-file suffix -> latest same-suffix canonical signal artifact.
effective=[]
for fp in glob.glob('factors/*.json'):
 try:
  d=json.load(open(fp))
  if d.get('validation',{}).get('status')=='EFFECTIVE':
   suffix=re.sub(r'^miner_\d+_\d{8}_','',os.path.basename(fp)[:-5])
   matches=glob.glob('scripts/*_'+suffix+'_signal.pkl')
   effective.append((os.path.basename(fp),suffix,sorted(matches)[-1] if matches else None))
 except Exception as e: print('JSON_ERROR',fp,type(e).__name__)
print('EFFECTIVE_COUNT',len(effective))
cor=[]; failures=[]
a=sig.stack().replace([np.inf,-np.inf],np.nan)
for fname,suffix,p in effective:
 try:
  if not p: raise ValueError('no_suffix_matched_signal')
  q=pickle.load(open(p,'rb'))
  if not isinstance(q,pd.DataFrame): raise ValueError('not_dataframe')
  q=q.stack().replace([np.inf,-np.inf],np.nan)
  z=pd.concat([a.rename('a'),q.rename('b')],axis=1).dropna()
  if len(z)<=100: raise ValueError('insufficient_finite_overlap_'+str(len(z)))
  r=spearmanr(z.a,z.b).statistic
  if not np.isfinite(r): raise ValueError('nonfinite_rho')
  cor.append((fname,os.path.basename(p),r,len(z)))
 except Exception as e: failures.append((fname,str(e)))
print('AUDIT_COMPARED',len(cor),'AUDIT_FAILURES',len(failures))
print('AUDIT_FAILURE_DETAIL',failures)
print('MAX_LIBRARY_CORR',max(cor,key=lambda x:abs(x[2])) if cor else None)
print('TOP_LIBRARY_CORR',[(x[0],round(x[2],5),x[3]) for x in sorted(cor,key=lambda x:abs(x[2]),reverse=True)[:8]])
