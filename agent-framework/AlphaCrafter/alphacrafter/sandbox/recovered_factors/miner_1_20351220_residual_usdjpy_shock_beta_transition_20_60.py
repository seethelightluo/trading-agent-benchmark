"""Miner 1: residual USDJPY shock-beta transition, one interpretable candidate."""
import glob,json
import numpy as np,pandas as pd
END=pd.Timestamp('2035-12-19'); ROOT='../persistent/stock_data'; IROOT='../persistent/index_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(path):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date'); return d['close'].replace([np.inf,-np.inf],np.nan)
px=pd.DataFrame({a:close(f'{ROOT}/{a}.csv').loc[:END] for a in assets}).sort_index().ffill()
fx=close(f'{IROOT}/USDJPY.csv').loc[:END].reindex(px.index).ffill(); r=px.pct_change(); fr=fx.pct_change()
# Each asset's sensitivity to yen shocks, short minus long. Residualize trend to avoid a directional carry/momentum proxy.
def beta(w,minp): return r.rolling(w,min_periods=minp).cov(fr).div(fr.rolling(w,min_periods=minp).var(),axis=0)
b20,b60=beta(20,15),beta(60,45); raw=b20-b60; m20=px.pct_change(20);m60=px.pct_change(60)
sig=pd.DataFrame(np.nan,index=px.index,columns=assets)
for d in px.index:
 y=raw.loc[d]; X=pd.concat([m20.loc[d],m60.loc[d]],axis=1); ok=y.notna()&X.notna().all(axis=1)
 if ok.sum()>=8:
  z=np.c_[np.ones(ok.sum()),X.loc[ok].to_numpy()];sig.loc[d,ok]=y.loc[ok]-z@np.linalg.lstsq(z,y.loc[ok],rcond=None)[0]
def calc(s,e,h):
 fw=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.loc[s:e].index:
  q=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=np.array(vals);return {'dates':len(a),'ic':float(a.mean()) if len(a) else None,'icir':float(a.mean()/a.std(ddof=1)) if len(a)>1 else None,'hit':float((a>0).mean()) if len(a) else None,'mean_instruments':float(np.mean(ns)) if ns else None}
metrics={str(h)+'d':calc('2020-01-01',END,h) for h in [1,5,10,20]}
# Exact panel evidence: admitted factor JSON names are matched to corresponding signal files; if absent, reject admission.
active=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f));
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append((f,j['factor_id']))
 except: pass
cors=[];missing=[]
for f,fid in active:
 key=fid.replace('miner_1_','miner_1_').replace('miner_2_','miner_2_').replace('miner_3_','miner_3_')
 matches=glob.glob('scripts/*'+key.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*signal.pkl')
 if not matches: missing.append(fid);continue
 best=None
 for p in matches:
  try:
   z=pd.read_pickle(p);z=z.unstack() if isinstance(z,pd.Series) else z;z.index=pd.to_datetime(z.index)
   ii=sig.index.intersection(z.index);cc=sig.columns.intersection(z.columns);q=pd.concat([sig.loc[ii,cc].stack(),z.loc[ii,cc].stack()],axis=1).dropna()
   if len(q)>=100:
    v=(abs(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')),fid,len(q),p)
    if best is None or v[0]>best[0]:best=v
  except Exception: pass
 if best is None:missing.append(fid)
 else:cors.append(best)
cors.sort(reverse=True)
rank=sig.rank(axis=1,pct=True)
out={'candidate':'residual_usdjpy_shock_beta_transition_20_60','cutoff':str(END.date()),'signal_dates':int(sig.notna().any(axis=1).sum()),'valid_cells':int(sig.notna().sum().sum()),'coverage':float(sig.notna().mean().mean()),'avg_instruments_per_signal_date':float(sig.notna().sum(axis=1).mean()),'turnover':float(rank.diff().abs().stack().mean()),'metrics':metrics,'regimes':{n:calc(a,b,10) for n,a,b in [('2020-2024','2020-01-01','2024-12-31'),('2025-2029','2025-01-01','2029-12-31'),('2030-2034','2030-01-01','2034-12-31'),('2035YTD','2035-01-01',END)]},'library_factors_effective':len(active),'library_correlation_evidence_complete':not bool(missing),'missing_library_signal_evidence':missing,'max_abs_library_correlation':({'rho':cors[0][0],'factor_id':cors[0][1],'common_cells':cors[0][2],'signal_path':cors[0][3]} if cors else None)}
print(json.dumps(out,indent=2));sig.to_pickle('scripts/miner_1_20351220_residual_usdjpy_shock_beta_transition_20_60_signal.pkl')
json.dump(out,open('scripts/miner_1_20351220_residual_usdjpy_shock_beta_transition_20_60_results.json','w'),indent=2)
