"""Validate one idea: residual crypto-upside transmission contraction (20d versus 60d)."""
import json, numpy as np, pandas as pd
# Point-in-time loader, market residualization, and admitted-library signal reconstruction.
src=open('scripts/miner_2_20291004_residual_crypto_basket_correlation_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: residual correlation')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-09-19')", "END=pd.Timestamp('2029-10-31')")
exec(prefix,globals())
# Candidate: change in loading on continuous positive BTC/ETH idiosyncratic shocks.
# High score means recent upside-crypto transmission has contracted versus 60d structure.
crypto=e[['BTC','ETH']].mean(axis=1)
crypto_up=crypto.clip(lower=0)**2
b20=pd.DataFrame({a:load(a,20,14,crypto_up) for a in A})
b60=pd.DataFrame({a:load(a,60,42,crypto_up) for a in A})
f=b60-b20
print('FACTOR residual_crypto_upside_transmission_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(out)); ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_28',(ics[10].index>='2027')&(ics[10].index<'2029')),('2029_ytd',ics[10].index>='2029')]:
 x=ics[10][mask]; print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))
