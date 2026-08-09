"""Miner 1 exploration: VIX-elevated 10-day reversal, all inputs capped at 2035-09-12."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-09-12'); FID='miner_1_vix_elevated_cross_asset_reversal_10v20obs'
def cl(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:cl('../persistent/stock_data/'+a+'.csv') for a in A}).loc[:END]
vix=cl('../persistent/index_data/VIX.csv').reindex(C.index).ffill(); r=np.log(C).diff()
# A short, interpretable corrective signal: fade 10d relative winners only during above-average VIX.
raw=-r.rolling(10,min_periods=10).sum(); state=vix>vix.rolling(20,min_periods=15).mean()
F=raw.where(state,0.0)
def calc(h):
 y=C.shift(-h).div(C).sub(1); out=[]; nn=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));nn.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(nn))}
M={}
for h in [1,5,10,20,40]:
 _,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=calc(10)
for tag,lo,hi in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_2035','2034-01-01',str(END.date()))]:
 z=x.loc[lo:hi]; print('REGIME_10D',tag,'dates',len(z),'IC',float(z.mean()),'ICIR',float(z.mean()/z.std(ddof=1)) if len(z)>1 else None,'hit',float((z>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
E={}; mx=0.; most=None; complete=True
for fid in active:
 key=fid
 ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps: E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 try:
  L=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A); z=pd.concat([F.stack(),L.stack()],axis=1).dropna();q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 except: q=np.nan;z=pd.DataFrame()
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);most=fid
print('PANEL',F.index.min().date(),END.date(),'signal_dates',int(F.ne(0).any(axis=1).sum()),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'state_rate',float(state.mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('DECAY',json.dumps(M,sort_keys=True)); print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active)); print('EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_1_20350913_vix_elevated_cross_asset_reversal_10v20obs_signal.pkl')
