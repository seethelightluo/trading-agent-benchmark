"""Clean standalone validation: continuous US-CN yield-level-spread innovation loading contraction (20d vs 60d).
Uses the intersection trading calendar, so no inherited/outer-calendar rolling-window alignment.
Cutoff is the last completed bar before 2031-08-07.
"""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-08-06')
def load(a,c='close'):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()[c].astype(float)
# Global completed-session intersection prevents weekend/holiday gaps from corrupting rate driver windows.
p=pd.concat({a:load(a) for a in A},axis=1,join='inner').loc[:END]
vol=pd.concat({a:load(a,'volume') for a in A},axis=1,join='inner').loc[:END]
r=p.pct_change(); m=r.mean(axis=1)
def rollbeta(x,w,n):
 return pd.DataFrame({a:r[a].rolling(w,min_periods=n).cov(x)/(x.rolling(w,min_periods=n).var()+1e-12) for a in A})
# Yield *level* spread avoids percentage-return instability for rates. Standardize its daily innovation.
spread=p.US10Y-p.CN10Y
delta=spread.diff()
driver=((delta-delta.rolling(60,min_periods=40).mean())/(delta.rolling(60,min_periods=40).std()+1e-12)).clip(-5,5)
f=rollbeta(driver,60,42)-rollbeta(driver,20,14)
print('FACTOR clean_us_cn_yield_level_spread_innovation_loading_contraction_20_60d','end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'dates',len(p),'universe',len(A),'driver_coverage',round(driver.notna().mean(),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask]; sd=x.std(ddof=1); print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns),'VALID_CELLS',int(f.notna().sum().sum()))
print('DECAY',json.dumps({str(h):{'ic':round(float(x['daily_paper_ic']),6),'icir':round(float(x['daily_paper_icir']),6),'dates':x['ic_dates']}for h,x in metrics.items()}))
# This standalone test deliberately does not claim library-correlation evidence; persistence requires a separate exhaustive library screen.
