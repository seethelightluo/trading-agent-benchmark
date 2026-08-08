"""Miner 1: VIX-normalization cross-sectional downside-asymmetry reversal, through 2035-12-05."""
import os,glob,json,re
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-12-05')
def px(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
def ix(a):
 d=pd.read_csv('../persistent/index_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:px(a) for a in A}).loc[:END]; R=C.pct_change()
vix=ix('VIX').reindex(C.index).ffill()
# When volatility normalizes after being elevated, favor assets with unusually downside-skewed
# recent paths: a conservative mean-reversion/recovery signal, normalized asset by asset.
down=R.clip(upper=0).pow(2).rolling(20,min_periods=20).mean()
up=R.clip(lower=0).pow(2).rolling(20,min_periods=20).mean()
base=down/(up+down).replace(0,np.nan)
normalizing=(vix<vix.rolling(20,min_periods=15).mean()) & (vix.rolling(60,min_periods=45).mean()>vix.rolling(252,min_periods=180).median())
F=base.where(normalizing,0.0)
F.to_pickle('scripts/miner_1_20351206_vix_normalization_downside_asymmetry_reversal_20x20x60obs_signal.pkl')
def metric(h):
 y=C.shift(-h).div(C)-1; out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,q));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in [1,5,10,20,40]:
 x,M[h]=metric(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=metric(10)
for tag,lo,hi in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_recent','2034-01-01',str(END.date()))]:
 z=x.loc[lo:hi];print('REGIME',tag,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None,float((z>0).mean()) if len(z) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
E={};mx=0;most=None;complete=True
for fid in active:
 suffix=re.sub(r'^miner_\d+_\d{8}_','',fid); ps=glob.glob('scripts/*_'+suffix+'_signal.pkl')
 if not ps:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 try:
  Q=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('f'),Q.stack().rename('q')],axis=1).dropna();q=spearmanr(z.f,z.q).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);most=fid
print('PANEL',F.index.min().date(),END.date(),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(1).mean()),'activation',float(normalizing.mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('DECAY',json.dumps(M,sort_keys=True));print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active));print('EVIDENCE',json.dumps(E,sort_keys=True))
