"""miner_2: relative realized-volatility shock, 10d / 60d, cross-sectionally ranked."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-12-25')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# A volatility-shock factor: recent realized risk relative to each asset's own baseline.
F=(R.rolling(10,min_periods=8).std()/R.rolling(60,min_periods=45).std()).loc[:END]
def metric(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[]; ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): out.append((d,float(q))); ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
res={}
for direction,X in [('shock',F),('inverse_shock',-F)]:
 for h in [1,5,10,20]:
  s,m=metric(X,h);res[direction+'_'+str(h)]=m;print('HORIZON',direction,h,json.dumps(m,sort_keys=True))
  if h==10:
   for lab,yrs in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030])]:
    q=s[s.index.year.isin(yrs)]; print('REGIME',direction,lab,json.dumps({'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit_ratio':float((q>0).mean()) if len(q) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  x=json.load(open(p));
  if x.get('validation',{}).get('status')=='EFFECTIVE':active.append(x['factor_id'])
 except:pass
complete=True; mx=0.;who=None
for fid in active:
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if fid in os.path.basename(p)]
 if not paths: complete=False;print('LIBRARY_CORR',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=float(spearmanr(z.x,z.l).statistic) if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 print('LIBRARY_CORR',fid,len(z),q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
print('SUMMARY',json.dumps({'factor':'relative_realized_volatility_shock_10v60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'observed_max':mx,'most_correlated':who,'metrics':res},sort_keys=True))
F.to_pickle('scripts/miner_2_20301226_relative_realized_volatility_shock_10v60obs_signal.pkl')
