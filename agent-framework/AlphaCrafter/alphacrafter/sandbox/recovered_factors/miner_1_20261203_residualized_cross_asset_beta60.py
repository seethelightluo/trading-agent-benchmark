"""miner_1: 60d cross-asset market-beta residual, one interpretable defensive candidate."""
import json, glob, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-12-02'
p={};v={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 p[a]=d.close.astype(float);v[a]=d.volume.astype(float)
p=pd.DataFrame(p); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); mkt=r.mean(axis=1)
# One idea: low 60d beta to the equal-weighted tradable cross-asset market, residualized CS against own volatility.
beta=pd.DataFrame({a:-r[a].rolling(60,min_periods=45).cov(mkt)/mkt.rolling(60,min_periods=45).var() for a in A})
def resid(y,x):
 z=pd.concat([y.rename('y'),x.rename('x')],axis=1).dropna(); out=pd.Series(np.nan,index=A)
 if len(z)>=8 and z.x.std()>0:
  X=np.c_[np.ones(len(z)),z.x];out.loc[z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
f=pd.DataFrame([resid(beta.iloc[i],vol.iloc[i]) for i in range(len(p))],index=p.index)
# Reconstruct all currently admitted factor signals for mandatory independence screen.
trend=(p/p.shift(20)-1)/vol
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
vol5=r.rolling(5,min_periods=4).std()
rvp=np.log(pd.DataFrame(v)/pd.DataFrame(v).rolling(20,min_periods=15).mean())
vov=vol5.rolling(20,min_periods=15).std()/vol5.rolling(20,min_periods=15).mean()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].pct_change()
vb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A})
vx=pd.DataFrame([resid(vb.iloc[i],vol.iloc[i]) for i in range(len(p))],index=p.index)
# admitted miner_2 downside beta improvement: beta down days, recent 20d minus prior 60d (sign convention defensive improvement)
down=mkt.where(mkt<0)
def db(x,w): return x.where(mkt<0).rolling(w,min_periods=max(12,int(w*.7))).cov(down)/down.rolling(w,min_periods=max(12,int(w*.7))).var()
dbimp=pd.DataFrame({a:-(db(r[a],20)-db(r[a],60)) for a in A})
lib={'risk_adjusted_trend_20d':trend,'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'relative_volume_participation_20d':rvp,'vol_of_vol_cv20':vov,'residualized_vix_stress_resilience_beta20':vx,'downside_beta_improvement_120_20':dbimp}
print('FACTOR residualized_cross_asset_market_beta60; END',END,'universe',len(A),'panel',p.index.min().date(),p.index.max().date())
allm={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:out.append((dt,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 ic=pd.Series(dict(out)); sd=ic.std(ddof=1); ranks=f.rank(axis=1,pct=True); turns=[]
 for i in range(1,len(ranks)):
  z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 M={'daily_paper_ic':ic.mean(),'daily_paper_icir':ic.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(ic)),'ic_hit_ratio':(ic>0).mean(),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':np.mean(ns),'mean_cross_sectional_coverage':np.mean(ns)/15,'mean_rank_turnover':np.mean(turns)};allm[h]=M
 print('HORIZON',h,json.dumps(M,default=float))
 for nm,mask in [('2020',ic.index<'2021-01-01'),('2021_2022',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_2024',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_2026',ic.index>='2025-01-01')]:
  q=ic[mask];print('REGIME',h,nm,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('SIGNAL_CELL_COVERAGE',f.notna().mean().mean(),'cells',int(f.notna().sum().sum()),'of',f.size)
mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',rho,'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'library_json_count',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps(allm,default=float))
