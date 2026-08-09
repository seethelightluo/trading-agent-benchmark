"""One factor idea: DXY impulse exposure activated when DXY volatility is elevated.
Data and signal are truncated at 2031-06-25, the completed day before runtime."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-06-25')
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
dxy=load('../persistent/index_data/DXY.csv').reindex(C.index).ffill()
r=np.log(C).diff(); dr=np.log(dxy).diff()
# A 40-observation asset exposure, multiplied by a 5-observation dollar impulse.
# State activates only when current 10d DXY realized volatility exceeds its prior 40d mean.
beta=r.rolling(40,min_periods=30).cov(dr).div(dr.rolling(40,min_periods=30).var(),axis=0)
dxyvol=dr.rolling(10,min_periods=8).std()
state=dxyvol.gt(dxyvol.rolling(40,min_periods=30).mean())
F=beta.mul(-dr.rolling(5,min_periods=5).sum(),axis=0).where(state,np.nan).loc[:END]
def ev(h):
 y=(C.shift(-h)/C-1).reindex(F.index); out=[]; nn=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,q));nn.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(nn))}
M={};S={}
for h in (1,5,10,20): S[h],M[h]=ev(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for n,mask in [('2020_2021',S[10].index.year<=2021),('2022_2023',S[10].index.year.isin([2022,2023])),('2024_2026',S[10].index.year.isin([2024,2025,2026])),('2027_2030',S[10].index.year.isin([2027,2028,2029,2030])),('2031_ytd',S[10].index.year==2031)]:
 s=S[10][mask];print('REGIME10',n,'dates',len(s),'ic',float(s.mean()) if len(s) else None,'icir',float(s.mean()/s.std(ddof=1)) if len(s)>1 else None,'hit',float((s>0).mean()) if len(s) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
# Required correlation audit of every effective persisted factor.
own='miner_3_dxy_vol_state_impulse_exposure_5v40v10x40obs'; complete=True; evidence={};mx=0;who=None;mcells=0
for p in glob.glob('factors/*.json'):
 d=json.load(open(p));fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE' or fid==own:continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 pp=[x for x in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(x)]
 if not pp: evidence[fid]={'rho':None,'cells':0};complete=False;print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(pp,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'cells':len(z)};complete &=rho is not None
 if rho is not None and abs(rho)>mx:mx=abs(rho);who=fid;mcells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
print('SUMMARY',json.dumps({'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'state_active_fraction':float(state.loc[F.index].mean()),'rank_stability_1d':float(np.mean(st)),'turnover_proxy':float(1-np.mean(st)),'metrics':M,'max_abs_library_correlation':mx,'most_correlated':who,'common_cells_most':mcells,'library_evidence_complete':complete,'library_evidence':evidence},sort_keys=True))
F.to_pickle('scripts/miner_3_20310626_dxy_vol_state_impulse_exposure_5v40v10x40obs_signal.pkl')
