"""Miner 2 — inverse USDCNY-upside beta resilience, single macro-conditional candidate."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-02-06')
P={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.where(d.close>0)
px=pd.DataFrame(P); ar=px.pct_change()
md=pd.read_csv('../persistent/index_data/USDCNY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
mc=md['close'].where(md['close']>0).pct_change().reindex(px.index)
# Negative beta to positive-USDCNY observations: high score means resilience when CNY weakens.
# Covariance is estimated only over the latest 40 days with at least 8 positive macro moves.
up=mc.where(mc>0)
def beta(s):
 ok=s.notna() & up.notna()
 if ok.sum()<8:return np.nan
 x=up[ok]; return -np.cov(s[ok],x,ddof=1)[0,1]/np.var(x,ddof=1) if np.var(x,ddof=1)>0 else np.nan
f=ar.rolling(40,min_periods=20).apply(lambda x: beta(pd.Series(x,index=ar.index[:len(x)])),raw=False)
# rolling.apply receives incorrect date labels above, so recompute directly to preserve macro alignment
f=pd.DataFrame(index=px.index,columns=ASSETS,dtype=float)
for t in range(39,len(px)):
 x=up.iloc[t-39:t+1]
 for a in ASSETS:
  y=ar[a].iloc[t-39:t+1];ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].var(ddof=1)>0:f.loc[px.index[t],a]=-np.cov(y[ok],x[ok],ddof=1)[0,1]/x[ok].var(ddof=1)
print('FACTOR inverse_usdcny_upside_beta_40obs visible_through',END.date(),'panel_dates',len(f),'assets',len(ASSETS))
print('SIGNAL_COVERAGE %.2f%% MEAN_VALID %.2f'%(100*f.notna().mean().mean(),f.notna().sum(axis=1).mean()))
def stat(q):
 if len(q)<2:return len(q),np.nan,np.nan,np.nan
 return len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()
def run(h):
 fw=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8 and f.loc[dt,ok].nunique()>1:rows.append((dt,spearmanr(f.loc[dt,ok],fw.loc[dt,ok]).statistic,ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']);n,ic,ir,hit=stat(z)
 print('\nHORIZON',h,'IC_DATES',n,'MEAN_NAMES %.2f'%(z.n.mean() if n else np.nan),'IC %.5f ICIR %.5f HIT %.2f%%'%(ic,ir,100*hit))
 for lab,lo,hi in [('2020-2022','2020-01-01','2022-12-31'),('2023-2024','2023-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027-2028','2027-01-01','2028-12-31'),('2029-2030','2029-01-01','2030-02-06')]:
  q=z[(z.date>=lo)&(z.date<=hi)];a,b,c,d=stat(q);print(lab,'dates',a,'IC',f'{b:.5f}' if a else 'NA','ICIR',f'{c:.5f}' if a else 'NA','hit',f'{100*d:.2f}%' if a else 'NA')
 valid=f.dropna(thresh=8); ss=[]
 for i in range(1,len(valid)):
  ok=valid.iloc[i].notna()&valid.iloc[i-1].notna()
  if ok.sum()>=8:ss.append(spearmanr(valid.iloc[i][ok],valid.iloc[i-1][ok]).statistic)
 s=np.nanmean(ss);print('RANK_STABILITY %.5f IMPLIED_TURNOVER %.2f%%'%(s,100*(1-s)/2))
for h in [1,5,10,20]:run(h)
# Required library audit: artifact lookup is exact only where a prior signal panel exists.
eff=[]
for fn in glob.glob('factors/*.json'):
 d=json.load(open(fn));
 if d.get('validation',{}).get('status')=='EFFECTIVE':eff.append(d.get('factor_id'))
art=glob.glob('scripts/*_signal.pkl'); available=[];cors=[]
for p in art:
 try:
  q=pd.read_pickle(p).reindex(index=f.index,columns=ASSETS)
  vals=[]
  for dt in f.index:
   ok=f.loc[dt].notna()&q.loc[dt].notna()
   if ok.sum()>=8: vals.append(abs(spearmanr(f.loc[dt,ok],q.loc[dt,ok]).statistic))
  if vals: available.append(os.path.basename(p));cors.append((np.nanmax(vals),os.path.basename(p)))
 except Exception:pass
print('\nLIBRARY_AUDIT effective',len(eff),'recoverable_artifacts',len(available),'missing_evidence',len(eff)-len(available))
print('MAX_ABS_AVAILABLE_CORR',max(cors) if cors else 'NA')
f.to_pickle('scripts/miner_2_20300207_inverse_usdcny_upside_beta_40obs_signal.pkl')
