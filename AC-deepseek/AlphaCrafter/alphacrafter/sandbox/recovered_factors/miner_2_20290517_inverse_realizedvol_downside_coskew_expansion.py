"""Validate one idea: inverse realized-volatility-stress residual downside co-skewness transition."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290419_realizedvol_stress_residual_upside_coskewness_contraction_20_60d.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction in residual loading')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-04-18')", "END=pd.Timestamp('2029-05-16')")
exec(prefix,globals())
# Add all factors admitted after the base library, so the novelty screen is binding.
rv=r.std(axis=1,ddof=0); stress_med=rv>rv.rolling(60,min_periods=40).median()
up_med=(market.where((market>0)&stress_med,0.0)**2)
lib['miner_3_realizedvol_stress_residual_upside_coskewness_contraction_20_60d']=pd.DataFrame({a:-(load(a,20,14,up_med)-load(a,60,42,up_med)) for a in A})
dlag=(1-p.shift(1)/p.shift(1).rolling(10,min_periods=8).max()).clip(lower=0)
def cap(w,n): return (e.clip(lower=0)*dlag).rolling(w,min_periods=n).mean()/(dlag.rolling(w,min_periods=n).mean()+1e-12)
lib['miner_2_contrarian_residual_drawdown_conditioned_upside_capture_acceleration_20_60d']=-(cap(20,14)-cap(60,42))
# Inverse orientation of rejected 2029-05-03 construction: expansion, not contraction,
# in residual loading on squared broad downside shocks during endogenous turbulence.
downshock=(market.where((market<0)&stress_med,0.0)**2)
f=pd.DataFrame({a:(load(a,20,14,downshock)-load(a,60,42,downshock)) for a in A})
print('FACTOR inverse_realizedvol_stress_residual_downside_coskewness_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'stress_day_fraction',round(float(stress_med.mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append((t,q));ns.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_onward',ics[10].index>='2027')]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))
