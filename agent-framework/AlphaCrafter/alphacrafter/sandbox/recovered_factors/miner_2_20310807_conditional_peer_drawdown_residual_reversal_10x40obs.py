"""miner_2 single-idea study: conditional peer-drawdown residual reversal, 10x40 observations."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-08-06')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# At each day calculate leave-one-out peer returns. A peer drawdown is a negative
# trailing 10-session peer return known at the prior close. The signal negates the
# asset's accumulated residual performance during those historical drawdown states:
# relative underperformance in stress is hypothesized to rebound over the next block.
P=pd.DataFrame({a:R.drop(columns=a).median(axis=1) for a in A})
E=R-P
peer10=pd.DataFrame({a:P[a].rolling(10,min_periods=8).sum() for a in A})
state=(peer10.shift(1)<0).astype(float)
# Require at least 10 stress-state sessions in the 40-session history, and use a
# conditional mean rather than a raw sum so different asset-class calendars/counts do not drive ranks.
num=(E*state).rolling(40,min_periods=20).sum(); den=state.rolling(40,min_periods=20).sum()
F=-(num/den.where(den>=10)).loc[:END]
def met(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[];ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,q));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float);s.index=pd.to_datetime(s.index);sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in (1,5,10,20):
 s,m=met(F,h);print('HORIZON',h,json.dumps(m,sort_keys=True),flush=True)
 if h==10:
  for lab,yrs in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_YTD',[2031])]:
   q=s[s.index.year.isin(yrs)];print('REGIME',lab,len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,float((q>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception:pass
mx=0;who=None;complete=True;ev={}
for fid in active:
 key=fid.split('_',2)[-1];cand=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not cand:complete=False;ev[fid]={'rho':None,'common_signal_cells':0};continue
 p=max(cand,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor':'conditional_peer_drawdown_residual_reversal_10x40obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_2_20310807_conditional_peer_drawdown_residual_reversal_10x40obs_signal.pkl')
