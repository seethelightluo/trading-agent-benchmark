"""Miner 2 revalidation: correlation-regime inverse tail skew through 2034-04-12."""
import os,glob,json,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-04-12'); FID='miner_2_corr_regime_inverse_tail_skew_20v60obs'
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END];R=C.pct_change(); mu=R.rolling(20,min_periods=16).mean();sd=R.rolling(20,min_periods=16).std().replace(0,np.nan); S=((R-mu)/sd).rolling(20,min_periods=16).skew()
peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A}); cor=pd.DataFrame({a:R[a].rolling(20,min_periods=16).corr(peer[a]) for a in A});F=(-S).where(cor>cor.rolling(60,min_periods=45).median())
def ev(h):
 y=C.shift(-h)/C-1;out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((dt,q));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float);v=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/v),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(v/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 s,M[h]=ev(h);print('HORIZON',h,json.dumps(M[h]))
s,_=ev(20)
for name,years in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('recent_2032_2034',[2032,2033,2034])]:
 x=s[s.index.year.isin(years)];v=x.std(ddof=1); print('REGIME_20D',name,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/v) if len(x)>1 and v else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
E={};mx=0;most=None;complete=True
for p in glob.glob('factors/*.json'):
 j=json.load(open(p)); fid=j.get('factor_id');
 if j.get('validation',{}).get('status')!='EFFECTIVE' or fid==FID:continue
 key=fid
 for pre in ['miner_1_','miner_2_','miner_3_']:key=key.replace(pre,'')
 ps=[x for x in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(x)]
 if not ps:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 try:
  q=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A); z=pd.concat([F.stack(),q.stack()],axis=1).dropna();rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();rho=np.nan
 E[fid]={'rho':float(rho) if np.isfinite(rho) else None,'common_signal_cells':len(z)}
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(float(rho));most=fid
print('PANEL',F.index.min().date(),END.date(),'dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('LIBRARY',mx,most,'COMPLETE',complete);print('EVIDENCE',json.dumps(E));F.to_pickle('scripts/miner_2_20340413_corr_regime_inverse_tail_skew_20v60obs_revalidation_signal.pkl')
