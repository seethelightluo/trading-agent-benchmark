"""Miner 3 research: cross-commodity downside-volatility exposure, information through 2031-02-19."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-02-19')
def get(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:get(a) for a in A}).sort_index(); R=np.log(C).diff()
# Idea: industrial-versus-energy downside-volatility divergence is a stress type; assets with historical
# sensitivity to this divergence are ranked only when the current divergence is unusually large.
down_c=R['COPPER'].clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
down_w=R['WTI'].clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
q=(down_c-down_w)/(down_c+down_w).replace(0,np.nan)
dq=q.diff()
beta=R.rolling(60,min_periods=45).cov(dq).div(dq.rolling(60,min_periods=45).var(),axis=0)
# Current 20d divergence level is signed; beta x level supplies a transparent conditional exposure score.
F=beta.mul(q).loc[:END]
def metrics(h):
 y=(C.shift(-h)/C-1).reindex(F.index); obs=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   rho=spearmanr(z.f,z.y).statistic
   if np.isfinite(rho):obs.append((d,float(rho)));widths.append(len(z))
 ic=pd.Series(dict(obs)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
M={}
for h in [1,5,10,20]:
 ic,M[h]=metrics(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=metrics(10)
for name,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year.isin([2027,2028,2029,2030])),('2031_ytd',ic.index.year==2031)]:
 x=ic[mask]; print('REGIME_10D',name,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v):st.append(v)
# Evidence against all admitted library factors; absence or undefined values is explicitly a failed gate.
evidence={}; mx=0.;most=None
for p in glob.glob('factors/*.json'):
 try:j=json.load(open(p))
 except:continue
 if j.get('validation',{}).get('status')=='DEPRECATED':continue
 fid=j.get('factor_id',os.path.basename(p)); hits=glob.glob('scripts/*'+fid.replace('miner_3_','').replace('miner_2_','').replace('miner_1_','')+'*_signal.pkl')
 if not hits:
  evidence[fid]={'rho':None,'common_signal_cells':0};mx=np.inf;continue
 try:
  L=pd.read_pickle(max(hits,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna(); rho=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();rho=np.nan
 evidence[fid]={'rho':float(rho) if np.isfinite(rho) else None,'common_signal_cells':len(z)}
 if not np.isfinite(rho):mx=np.inf
 elif abs(rho)>mx:mx=abs(rho);most=fid
print('FACTOR commodity_downside_volatility_divergence_exposure_20v60obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'implied_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps(M,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_3_20310220_commodity_downside_volatility_divergence_exposure_20v60obs_signal.pkl')
