"""miner_2: cross-asset residual disagreement reversal, 5 observations.
An asset's five-day return residual versus the contemporaneous universe median is
negated only when its residual direction has low cross-asset breadth (a
idiosyncratic/disagreed move).  The magnitude is volatility normalized. Values
at date d use information through d only; forward returns start after d.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-03-19')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index();R=C.pct_change();E=R.sub(R.median(axis=1),axis=0)
m=E.rolling(5,min_periods=5).mean(); vol=E.rolling(20,min_periods=20).std(ddof=1)
# For each date/sign, breadth is fraction of available assets sharing its 5d residual sign.
pos=(m>0).sum(axis=1); neg=(m<0).sum(axis=1); n=m.notna().sum(axis=1)
breadth=pd.DataFrame(np.where(m.gt(0),np.broadcast_to((pos/n).values[:,None],m.shape),np.where(m.lt(0),np.broadcast_to((neg/n).values[:,None],m.shape),np.nan)),index=m.index,columns=A)
F=(-m/vol*(1-breadth)).loc[:END].replace([np.inf,-np.inf],np.nan)
def measure(h):
 y=(C.shift(-h)/C-1).reindex(F.index); y.loc[y.index>C.loc[:END].index[-h-1]]=np.nan; rows=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q):rows.append((d,float(q)));widths.append(len(z))
 ic=pd.Series(dict(rows));sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
metrics={}
for h in (1,5,10,20):
 ic,x=measure(h);metrics[str(h)]=x;print('HORIZON',h,json.dumps(x,sort_keys=True))
 if h==5:
  for lab,yrs in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_ytd',[2031])]:
   v=ic[ic.index.year.isin(yrs)];print('REGIME_5D',lab,json.dumps({'dates':len(v),'ic':float(v.mean()) if len(v) else None,'icir':float(v.mean()/v.std(ddof=1)) if len(v)>1 else None,'hit':float((v>0).mean()) if len(v) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
def key(x):
 for p in ('miner_1_','miner_2_','miner_3_'):
  if x.startswith(p):return x[len(p):]
 return x
active=[]
for p in glob.glob('factors/*.json'):
 try:
  d=json.load(open(p));
  if d.get('validation',{}).get('status')=='EFFECTIVE':active.append(d['factor_id'])
 except:pass
complete=True;mx=0;who=None
for fid in active:
 ms=glob.glob('scripts/*'+key(fid)+'*signal.pkl')
 if not ms:complete=False;print('LIBRARY_CORR',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(ms,key=os.path.getmtime)).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna();q=float(spearmanr(z.a,z.b).statistic) if len(z)>=8 else np.nan
 except:q=np.nan;z=pd.DataFrame()
 print('LIBRARY_CORR',fid,'cells',len(z),'rho',q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
print('SUMMARY',json.dumps({'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'metrics':metrics},sort_keys=True))
F.to_pickle('scripts/miner_2_20310320_cross_asset_residual_disagreement_reversal_5obs_signal.pkl')
