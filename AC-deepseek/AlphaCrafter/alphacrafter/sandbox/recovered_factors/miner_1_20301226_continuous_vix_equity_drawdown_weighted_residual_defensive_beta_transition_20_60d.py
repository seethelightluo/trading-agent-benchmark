"""One candidate: continuous VIX/equity-drawdown weighted residual defensive-beta transition (20d vs 60d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-12-25')")
exec(prefix,globals())
# Smooth observed stress intensity: lagged VIX percentile times magnitude of completed equity drawdown.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill()
equity=p[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].pct_change().mean(axis=1)
equity_level=(1+equity.fillna(0)).cumprod()
eq_dd=equity_level/equity_level.rolling(20,min_periods=15).max()-1
vix_pct=vix.rolling(252,min_periods=126).rank(pct=True).shift(1)
stress=(vix_pct.clip(lower=.0)*(-eq_dd).clip(lower=0)).fillna(0)
# Normalize scale only; beta difference retains cross-sectional differences in residual defensive linkage.
stress=stress/stress.rolling(252,min_periods=126).mean().replace(0,np.nan)
defensive=e[['XAU','US10Y','CN10Y']].mean(axis=1)
f=beta(e,defensive*stress,20,14)-beta(e,defensive*stress,60,42)
print('FACTOR continuous_vix_equity_drawdown_weighted_residual_defensive_beta_transition_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_reconstructed',len(lib),'vix_coverage',round(vix.notna().mean(),6),'stress_availability',round(stress.notna().mean(),6),'mean_stress',round(stress.mean(),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 z=pd.Series(dict(out));ics[h]=z;sd=z.std(ddof=1)
 metrics[h]={'daily_paper_ic':z.mean(),'daily_paper_icir':z.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(z)),'ic_hit_ratio':(z>0).mean(),'ic_dates':len(z),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 z=ics[10][mask];print('REGIME10',name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION_RECONSTRUCTED',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']}for h,q in metrics.items()}))
