"""One-factor validation: intraday range-position persistence (20 sessions).
Signal is the trailing mean normalized close location within daily high-low range.
Persistent closing near a day's high/low measures buyer/seller control without using
absolute return magnitude, a distinct microstructure/path feature.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2033-08-17')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d
D={a:load(a) for a in A}; C=pd.DataFrame({a:D[a]['close'].astype(float) for a in A}).sort_index()
# close-location value is [-1,1], neutral if range unavailable or zero
P=pd.DataFrame({a:((D[a]['close']-D[a]['low'])/(D[a]['high']-D[a]['low'])*2-1).reindex(C.index) for a in A})
P=P.replace([np.inf,-np.inf],np.nan); F=P.rolling(20,min_periods=15).mean()
def ev(h):
 y=C.shift(-h).div(C)-1; o=[];n=[]
 for dt in F.index[:-h]:
  z=pd.concat([F.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):o.append((dt,q));n.append(len(z))
 s=pd.Series(dict(o));sd=s.std(ddof=1);return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(n))}
S={};M={}
for h in (1,5,10,20):S[h],M[h]=ev(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for lab,mask in [('2020_2022',S[5].index.year<=2022),('2023_2025',S[5].index.year.isin([2023,2024,2025])),('2026_2028',S[5].index.year.isin([2026,2027,2028])),('2029_2031',S[5].index.year.isin([2029,2030,2031])),('2032_2033',S[5].index.year>=2032)]:
 s=S[5][mask];print('REGIME5',lab,'dates',len(s),'ic',float(s.mean()),'icir',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
complete=True;mx=0.;peer=None;pcells=0;audited=0;evidence={}
for p in glob.glob('factors/*.json'):
 d=json.load(open(p));fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 audited+=1;key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','');paths=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not paths:complete=False;evidence[fid]={'rho':None,'cells':0};print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'cells':len(z)};complete &=rho is not None
 if rho is not None and abs(rho)>mx:mx=abs(rho);peer=fid;pcells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
summary={'factor_id':'miner_2_intraday_range_position_persistence_20obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'turnover_proxy':float(1-np.mean(st)),'metrics':M,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':peer,'common_cells_most':pcells,'library_evidence_complete':complete,'library_factors_compared':audited,'library_evidence':evidence}
print('SUMMARY',json.dumps(summary,sort_keys=True));F.to_pickle('scripts/miner_2_20330818_intraday_range_position_persistence_20obs_signal.pkl')
