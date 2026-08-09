"""Validate one idea: yield-volatility-conditioned US10Y shock-transmission exposure.
Uses only closes observable through 2031-07-23; forward returns are likewise truncated."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-07-23')
def load(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]
r=np.log(C).diff(); yr=r.US10Y
# A rate shock is informative only when its trailing 20d volatility exceeds its own 60d median.
# In that state, score assets by negative trailing 50d sensitivity times current 5d yield impulse.
beta=r.rolling(50,min_periods=35).cov(yr).div(yr.rolling(50,min_periods=35).var(),axis=0)
state=(yr.rolling(20,min_periods=20).std()>yr.rolling(60,min_periods=45).std()).astype(float)
F=beta.mul(-yr.rolling(5,min_periods=5).sum(),axis=0).mul(state,axis=0)
def ev(h):
 y=(C.shift(-h)/C-1).reindex(F.index); out=[]; ns=[]
 for d in F.index[:-h]:
  z=pd.concat([F.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,q));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
M={}; S={}
for h in (1,5,10,20):
 S[h],M[h]=ev(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for lab,mask in [('2020_2021',S[10].index.year<=2021),('2022_2023',S[10].index.year.isin([2022,2023])),('2024_2026',S[10].index.year.isin([2024,2025,2026])),('2027_2030',S[10].index.year.isin([2027,2028,2029,2030])),('2031_ytd',S[10].index.year==2031)]:
 s=S[10][mask]; print('REGIME10',lab,'dates',len(s),'ic',float(s.mean()),'icir',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna(); q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 if np.isfinite(q):st.append(q)
evidence={}; complete=True; mx=0.; peer=None; pcells=0
for p in glob.glob('factors/*.json'):
 d=json.load(open(p)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 # factor ids map to retained signal artifact by distinctive trailing name
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 paths=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not paths:
  print('LIB',fid,'MISSING');evidence[fid]={'rho':None,'cells':0};complete=False;continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna(); q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'cells':len(z)};complete &=rho is not None
 if rho is not None and abs(rho)>mx:mx=abs(rho);peer=fid;pcells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
summary={'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'state_active_fraction':float(state.mean()),'rank_stability_1d':float(np.mean(st)),'turnover_proxy':float(1-np.mean(st)),'metrics':M,'max_abs_library_correlation':mx,'most_correlated':peer,'common_cells_most':pcells,'library_evidence_complete':complete,'library_evidence':evidence}
print('SUMMARY',json.dumps(summary,sort_keys=True));F.to_pickle('scripts/miner_3_20310724_us10y_volstate_shock_transmission_5v50v20x60obs_signal.pkl')
