"""One-factor validation: residual downside-tail concentration expansion, inverted direction (20d vs 60d)."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-09-29')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float).loc[:END]
cal=pd.bdate_range('2020-01-01',END); p=pd.DataFrame({a:load(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); b=r.mean(axis=1)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(b)/(b.rolling(60,min_periods=42).var()+1e-12) for a in A}); e=r-beta.mul(b,axis=0)
def tailshare(w,mp):
 out=pd.DataFrame(index=e.index,columns=A,dtype=float); k=max(2,int(np.ceil(.2*w)))
 for a in A:
  z=e[a].to_numpy(); ans=np.full(len(z),np.nan)
  for i in range(w-1,len(z)):
   x=z[i-w+1:i+1]; x=x[np.isfinite(x)]
   if len(x)>=mp:
    neg=np.minimum(x,0.); den=(neg*neg).sum()
    if den>1e-16: ans[i]=(np.sort(neg)[:k]**2).sum()/den
  out[a]=ans
 return out
# Separate, explicitly inverted candidate: higher means recent downside residual shocks are MORE tail-concentrated than long-run state.
f=(tailshare(20,14)-tailshare(60,42)).shift(1)
print('FACTOR residual_downside_tail_concentration_expansion_20_60d_inverted')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1; obs=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.concat([f.loc[t].rename('f'),fw.loc[t].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v): obs.append((t,v));ns.append(len(q))
 x=pd.Series(dict(obs)); ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(x.mean(),6),'daily_paper_icir':round(x.mean()/sd,6),'ic_standard_error':round(sd/np.sqrt(len(x)),6),'ic_dates':len(x),'hit_ratio':round((x>0).mean(),6),'mean_valid_instruments':round(np.mean(ns),4)}))
for name,mask in [('2020_2024',ics[10].index<'2025-01-01'),('2025_2026',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask]; print('REGIME_10D',name,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
print('DECAY',json.dumps({str(k):round(v.mean(),6) for k,v in ics.items()}))
print('LIBRARY_SCREEN pending only if performance gates pass; signal-level comparison to every admitted factor is required for admission.')
