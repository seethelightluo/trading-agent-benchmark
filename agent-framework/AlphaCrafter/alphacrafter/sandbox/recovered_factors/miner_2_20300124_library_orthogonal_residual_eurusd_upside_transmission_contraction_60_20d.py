"""Validate one idea: residual EURUSD upside-transmission contraction (60d vs 20d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20291213_library_orthogonal_residual_dxy_downside_transmission_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# DXY downside shocks')[0].replace("END=pd.Timestamp('2029-12-12')", "END=pd.Timestamp('2030-01-23')")
exec(prefix,globals())
# Candidate: a narrowing residual response to EUR appreciation shocks. EURUSD is
# observation-only; factor is residualized against the separately admitted broad-USD transition.
eur=pd.read_csv('../persistent/index_data/EURUSD.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
shock=(eur.clip(lower=0))**2
def beta(a,w,n,x): return e[a].rolling(w,min_periods=n).cov(x)/x.rolling(w,min_periods=n).var()
raw=pd.DataFrame({a:-(beta(a,20,14,shock)-beta(a,60,42,shock)) for a in A})
# Reconstruct the admitted DXY transition signal for the required complete novelty screen.
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
absd=dxy.abs()
oldraw=-(pd.DataFrame({a:beta(a,20,12,absd) for a in A})-pd.DataFrame({a:beta(a,60,40,absd) for a in A}))
lib['miner_1_residualized_usd_shock_beta_transition_60_20d']=residual(oldraw,trend,own)
anchor='miner_1_residualized_usd_shock_beta_transition_60_20d'
def orth(t):
 z=pd.concat([raw.loc[t].rename('x'),lib[anchor].loc[t].rename('a')],axis=1).dropna()
 if len(z)<8 or z.a.nunique()<2:return pd.Series(index=A,dtype=float)
 return raw.loc[t]-(np.cov(z.x,z.a,ddof=1)[0,1]/np.var(z.a,ddof=1))*lib[anchor].loc[t]
f=pd.DataFrame({t:orth(t) for t in raw.index}).T
print('FACTOR library_orthogonal_residual_eurusd_upside_transmission_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'anchor',anchor)
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_28',(ics[10].index>='2027')&(ics[10].index<'2029')),('2029_onward',ics[10].index>='2029')]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))
