"""Miner 2 20300613: lower-tail frequency, a cross-asset fragility signal.
Tests whether assets with unusually frequent recent standardized downside shocks
underperform, or reverse, conditional only on their own trailing distribution."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-06-12')
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close(a) for a in A}); r=np.log(C).diff()
# At t, count shocks in the preceding 10 completed sessions. Scale each shock by
# a 40-session distribution known at the respective shock date; high means fragile.
z=(r-r.rolling(40,min_periods=30).mean())/r.rolling(40,min_periods=30).std(ddof=1)
F=(z<-1.0).rolling(10,min_periods=8).mean().loc[:END]
def getic(h):
 fw=(C.shift(-h)/C-1).reindex(F.index); rows=[]
 for dt in F.index:
  x=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(x)>=8 and x.f.nunique()>1:rows.append((dt,float(spearmanr(x.f,x.y).statistic),len(x)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); s=q.ic.std(ddof=1)
 return q,dict(ic_dates=len(q),mean_valid_instruments=float(q.n.mean()),daily_paper_ic=float(q.ic.mean()),daily_paper_icir=float(q.ic.mean()/s),ic_hit_ratio=float((q.ic>0).mean()),ic_standard_error=float(s/np.sqrt(len(q))))
ALL={}
for h in [1,5,10,20]:
 q,m=getic(h);ALL[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
q,_=getic(5)
for lab,lo,hi in [('2020_2021','2020-01-01','2021-12-31'),('2022_2023','2022-01-01','2023-12-31'),('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-06-12')]:
 x=q[(q.date>=lo)&(q.date<=hi)];sd=x.ic.std(ddof=1)
 print('REGIME_5D',lab,'dates',len(x),'IC',float(x.ic.mean()) if len(x) else None,'ICIR',float(x.ic.mean()/sd) if len(x)>1 else None,'hit',float((x.ic>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 x=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(x)>=8: st.append(float(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic))
active=[]
for fp in glob.glob('factors/*.json'):
 if fp.endswith('.bak') or '_deprecated' in fp:continue
 try:
  d=json.load(open(fp));
  if d.get('validation',{}).get('status')=='EFFECTIVE':active.append(d['factor_id'])
 except:pass
files=glob.glob('scripts/*_signal.pkl');ev={};mx=0.;who=None
for fid in active:
 key=fid
 matches=[p for p in files if key in os.path.basename(p)]
 if not matches:
  ev[fid]={'rho':None,'common_signal_cells':0,'file':None};mx=np.inf;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(matches,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A)
  x=pd.concat([F.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna();rho=float(spearmanr(x.candidate,x.library).statistic) if len(x)>=8 else np.nan
 except Exception: x=pd.DataFrame();rho=np.nan
 ev[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(x),'file':p}
 if not np.isfinite(rho):mx=np.inf
 elif abs(rho)>mx:mx=abs(rho);who=fid
 print('LIBRARY_CORR',fid,'cells',len(x),'spearman',rho)
print('FACTOR lower_tail_frequency_10v40obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who,'AUDITED',len(active),'EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20300613_lower_tail_frequency_10v40obs_signal.pkl')
