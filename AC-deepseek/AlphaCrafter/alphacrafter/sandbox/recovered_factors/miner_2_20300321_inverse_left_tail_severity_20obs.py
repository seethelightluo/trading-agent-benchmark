"""Miner 2 one-factor research: inverse left-tail severity, trailing 20 observations."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-03-20')
def close(sym,root):
 return pd.read_csv(root+'/'+sym+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].where(lambda x:x>0)
px=pd.DataFrame({a:close(a,'../persistent/stock_data') for a in A}); r=px.pct_change()
# Factor is the negative mean magnitude of daily returns below -1 standard deviation, normalized by total daily volatility.
# It distinguishes sustained/severe left-tail paths from merely volatile assets; higher is less recent left-tail severity.
f=pd.DataFrame(np.nan,index=px.index,columns=A)
for t in range(19,len(px)):
 w=r.iloc[t-19:t+1]
 for a in A:
  x=w[a].dropna()
  if len(x)>=15 and x.std(ddof=1)>0:
   downside=(-x[x < -x.std(ddof=1)]).mean() if (x < -x.std(ddof=1)).any() else 0.
   f.iloc[t,f.columns.get_loc(a)]=-downside/x.std(ddof=1)
print('FACTOR inverse_left_tail_severity_20obs visible_through',END.date(),'panel_dates',len(f),'assets',len(A))
print('SIGNAL_COVERAGE %.2f%% MEAN_VALID %.2f'%(100*f.notna().mean().mean(),f.notna().sum(1).mean()))
def stats(z):
 return (len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()) if len(z)>1 else (len(z),np.nan,np.nan,np.nan)
results={}
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8 and f.loc[dt,ok].nunique()>1: out.append((dt,spearmanr(f.loc[dt,ok],fw.loc[dt,ok]).statistic,ok.sum()))
 z=pd.DataFrame(out,columns=['date','ic','n']); n,ic,ir,hit=stats(z);results[h]=(n,ic,ir,hit)
 print('\nHORIZON',h,'IC_DATES',n,'MEAN_NAMES %.2f'%z.n.mean(),'IC %.5f ICIR %.5f HIT %.2f%%'%(ic,ir,100*hit))
 for lab,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029-30','2029-01-01','2030-03-20')]:
  q=z[(z.date>=lo)&(z.date<=hi)];nn,x,y,k=stats(q);print(lab,'dates',nn,'IC %.5f ICIR %.5f hit %.2f%%'%(x,y,100*k))
stab=[]
for i in range(1,len(f)):
 ok=f.iloc[i].notna()&f.iloc[i-1].notna()
 if ok.sum()>=8 and f.iloc[i][ok].nunique()>1 and f.iloc[i-1][ok].nunique()>1:stab.append(spearmanr(f.iloc[i][ok],f.iloc[i-1][ok]).statistic)
print('RANK_STABILITY %.5f IMPLIED_TURNOVER %.2f%%'%(np.mean(stab),50*(1-np.mean(stab))))
libs=[];missing=[]
for fn in glob.glob('factors/*.json'):
 try:d=json.load(open(fn))
 except:continue
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 desc='_'.join(d['factor_id'].split('_')[2:]);cand=glob.glob('scripts/*_'+desc+'_signal.pkl')
 if not cand: missing.append(d['factor_id']);continue
 try:libs.append((d['factor_id'],pd.read_pickle(cand[-1])))
 except:missing.append(d['factor_id'])
vals=[]
for fid,g in libs:
 z=pd.concat([f.stack().rename('x'),g.stack().rename('y')],axis=1).dropna()
 if len(z)>=8: vals.append((abs(spearmanr(z.x,z.y).statistic),fid,len(z)))
print('\nLIBRARY_AUDIT effective',len(libs)+len(missing),'loaded',len(libs),'missing',len(missing),missing)
print('LIBRARY_MAX',max(vals) if vals else 'NONE','COMPLETE',not missing)
f.to_pickle('scripts/miner_2_20300321_inverse_left_tail_severity_20obs_signal.pkl')
