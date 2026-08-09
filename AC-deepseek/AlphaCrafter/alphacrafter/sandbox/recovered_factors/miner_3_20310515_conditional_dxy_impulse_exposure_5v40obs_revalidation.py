"""Revalidate conditional DXY impulse exposure using information through 2031-05-14."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-05-14')
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index(); dxy=load('../persistent/index_data/DXY.csv').reindex(C.index).ffill()
r=np.log(C).diff(); dr=np.log(dxy).diff(); beta=r.rolling(40,min_periods=30).cov(dr).div(dr.rolling(40,min_periods=30).var(),axis=0)
F=beta.mul(-dr.rolling(5,min_periods=5).sum(),axis=0).loc[:END]
def test(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[]; n=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append((d,q));n.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,dict(daily_paper_ic=float(s.mean()),daily_paper_icir=float(s.mean()/sd),ic_hit_ratio=float((s>0).mean()),ic_standard_error=float(sd/np.sqrt(len(s))),ic_dates=len(s),mean_valid_instruments=float(np.mean(n)))
M={}; series={}
for h in [1,5,10,20]: series[h],M[h]=test(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for label,mask in [('2020_2021',series[20].index.year<=2021),('2022_2023',series[20].index.year.isin([2022,2023])),('2024_2026',series[20].index.year.isin([2024,2025,2026])),('2027_2030',series[20].index.year.isin([2027,2028,2029,2030])),('2031_ytd',series[20].index.year==2031)]:
 s=series[20][mask];print('REGIME20',label,len(s),float(s.mean()) if len(s) else None,float(s.mean()/s.std(ddof=1)) if len(s)>1 else None,float((s>0).mean()) if len(s) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and np.isfinite(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic):st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
own='miner_3_conditional_dxy_impulse_exposure_5v40obs'; ev={};complete=True;mx=0.;most=None
for p in glob.glob('factors/*.json'):
 d=json.load(open(p)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE' or fid==own:continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); paths=[x for x in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(x)]
 if not paths: ev[fid]={'rho':None,'cells':0};complete=False;print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;ev[fid]={'rho':rho,'cells':len(z)};complete &= rho is not None
 if rho is not None and abs(rho)>mx:mx=abs(rho);most=fid
 print('LIB',fid,'cells',len(z),'rho',rho)
print('SUMMARY',json.dumps({'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'turnover':float(1-np.mean(st)),'max_abs_library_correlation':mx,'most_correlated':most,'evidence_complete':complete,'metrics':M,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_3_20310515_conditional_dxy_impulse_exposure_5v40obs_revalidation_signal.pkl')
