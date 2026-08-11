import pandas as pd, numpy as np, glob, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close.astype(float)
px=pd.concat(P,axis=1).sort_index().loc[:'2027-10-20']
r=px.pct_change(fill_method=None)
# equal-weight market proxy, contemporaneous historical only
m=r.mean(axis=1,skipna=True)
# rolling beta from asset and market returns; factor uses through t only
factors=[]; fdates=[]; ics=[]; rows=[]
for t in range(80,len(px)-1):
 date=px.index[t]
 rr=r.iloc[t-59:t+1]; mm=m.iloc[t-59:t+1]
 vals={}
 for s in U:
  x=rr[s]; valid=x.notna()&mm.notna()
  if valid.sum()<30: continue
  beta=np.cov(x[valid],mm[valid],ddof=1)[0,1]/(np.var(mm[valid],ddof=1)+1e-12)
  resid5=(r[s].iloc[t-4:t+1].sum()-beta*m.iloc[t-4:t+1].sum())
  vol=x.std()*np.sqrt(252)
  if np.isfinite(vol) and vol>1e-8: vals[s]=-resid5/vol
 fr=pd.Series(vals).dropna()
 fut=r.iloc[t+1].reindex(fr.index)
 z=pd.concat([fr,fut.rename('y')],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.y,method='spearman'); ics.append(ic); fdates.append(date)
  rows += [{'date':date,'symbol':s,'signal':fr[s]} for s in fr.index]
# aggregate
v=np.array(ics); print('dates',len(v),'avg instruments',len(rows)/len(v),'IC',np.nanmean(v),'ICIR',np.nanmean(v)/(np.nanstd(v,ddof=1)+1e-12),'hit',np.mean(v>0))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=np.array([x for x,d in zip(ics,fdates) if a<=str(d.year)<=b]); print(a,b,len(q),np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12) if len(q)>1 else np.nan)
out=pd.DataFrame(rows); out.to_csv('scripts/miner_1_20271021_beta_resid_reversal_signal.csv',index=False)
print('coverage',out.symbol.nunique()/15,'rows',len(out))
