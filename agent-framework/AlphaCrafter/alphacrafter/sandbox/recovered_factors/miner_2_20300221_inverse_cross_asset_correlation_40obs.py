"""Miner 2: inverse cross-asset correlation beta, one interpretable diversification-resilience idea."""
import os,glob,json,re
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-02-20')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].where(lambda x:x>0) for a in A})
r=px.pct_change()
# For each asset, score is minus its trailing mean pairwise return correlation to the other 14 assets.
# Higher score identifies relatively independent assets, hypothesized to have better subsequent cross-asset returns.
f=pd.DataFrame(np.nan,index=px.index,columns=A)
for t in range(39,len(r)):
 w=r.iloc[t-39:t+1]
 c=w.corr(min_periods=25)
 for a in A:
  x=c.loc[a].drop(a).dropna()
  if len(x)>=8:f.loc[f.index[t],a]=-x.mean()
print('FACTOR inverse_cross_asset_correlation_40obs visible_through',END.date(),'panel_dates',len(f),'assets',len(A))
print('SIGNAL_COVERAGE %.2f%% MEAN_VALID %.2f'%(100*f.notna().mean().mean(),f.notna().sum(1).mean()))
def stats(z):
 return (len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1), (z.ic>0).mean()) if len(z)>1 else (len(z),np.nan,np.nan,np.nan)
allz={}
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8 and f.loc[dt,ok].nunique()>1:out.append((dt,spearmanr(f.loc[dt,ok],fw.loc[dt,ok]).statistic,ok.sum()))
 z=pd.DataFrame(out,columns=['date','ic','n']);allz[h]=z;n,ic,ir,hit=stats(z)
 print('\nHORIZON',h,'IC_DATES',n,'MEAN_NAMES %.2f'%(z.n.mean()),'IC %.5f ICIR %.5f HIT %.2f%%'%(ic,ir,100*hit))
 for lab,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029-30','2029-01-01','2030-02-20')]:
  q=z[(z.date>=lo)&(z.date<=hi)];n,x,y,k=stats(q);print(lab,'dates',n,'IC',f'{x:.5f}','ICIR',f'{y:.5f}','hit',f'{100*k:.2f}%')
# rank stability
v=f.dropna(thresh=8); q=[]
for i in range(1,len(v)):
 ok=v.iloc[i].notna()&v.iloc[i-1].notna()
 if ok.sum()>=8:q.append(spearmanr(v.iloc[i][ok],v.iloc[i-1][ok]).statistic)
print('RANK_STABILITY %.5f IMPLIED_TURNOVER %.2f%%'%(np.mean(q),100*(1-np.mean(q))/2))
# complete audit: exact artifact is expected to include date and factor-id suffix; report every missing explicitly
libs=[]; missing=[]
for fn in glob.glob('factors/*.json'):
 try:d=json.load(open(fn))
 except:continue
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 fid=d['factor_id']; stem=fid.replace('miner_','miner_')
 # scored signal files are named miner_N_YYYYMMDD_<suffix>_signal.pkl; identify using final descriptive segment
 desc='_'.join(fid.split('_')[2:]); cand=glob.glob('scripts/*_'+desc+'_signal.pkl')
 if not cand: missing.append(fid);continue
 try: libs.append((fid,pd.read_pickle(cand[-1])))
 except:missing.append(fid)
vals=[]
for fid,g in libs:
 x=f.stack().rename('x');y=g.stack().rename('y');z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: vals.append((abs(spearmanr(z.x,z.y).statistic),fid,len(z)))
print('\nLIBRARY_AUDIT effective',len(libs)+len(missing),'loaded',len(libs),'missing',len(missing),missing)
print('LIBRARY_MAX',max(vals) if vals else 'NONE','COMPLETE',not missing)
f.to_pickle('scripts/miner_2_20300221_inverse_cross_asset_correlation_40obs_signal.pkl')
