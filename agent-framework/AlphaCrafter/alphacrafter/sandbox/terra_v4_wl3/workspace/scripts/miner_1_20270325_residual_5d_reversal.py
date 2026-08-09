import os, json, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-24')
raw={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').sort_values('date').set_index('date')['close'].astype(float) for s in U}
prices=pd.DataFrame(raw).sort_index(); rets=prices.pct_change(); r3=-rets.rolling(3).sum()/rets.rolling(20).std(); r5=-rets.rolling(5).sum(); fac=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float)
for dt in prices.index:
 x=r3.loc[dt]; y=r5.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  xx=x[ok].values; yy=y[ok].values; vx=np.var(xx); beta=0 if vx<1e-12 else np.cov(xx,yy,bias=True)[0,1]/vx; fac.loc[dt,ok]=yy-(yy.mean()+beta*(xx-xx.mean()))
fwd=prices.shift(-1)/prices-1; ics=[]; rows=[]
for dt in prices.index[:-1]:
 ok=fac.loc[dt].notna()&fwd.loc[dt].notna()
 if ok.sum()>=8:
  z=fac.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
  if np.isfinite(z): ics.append(z); rows.append((dt,z,ok.sum()))
a=np.array(ics); ic=a.mean(); icir=ic/a.std(ddof=1)*np.sqrt(len(a)); ranks=fac.rank(axis=1,pct=True); turns=[]
for x,y in zip(ranks.index[:-1],ranks.index[1:]):
 ok=ranks.loc[x].notna()&ranks.loc[y].notna()
 if ok.sum()>=8: turns.append(np.abs(ranks.loc[x,ok]-ranks.loc[y,ok]).mean())
print(json.dumps({'validation_end':str(cut.date()),'dates':len(a),'avg_n':float(np.mean([x[2] for x in rows])),'coverage':float(fac.notna().sum().sum()/(len(fac)*15)),'ic':float(ic),'icir':float(icir),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))},indent=2))
for name,lo,hi in [('2020-22','2020-01-01','2023-01-01'),('2023-24','2023-01-01','2025-01-01'),('2025-27','2025-01-01','2027-03-25')]:
 z=np.array([v for d,v,n in rows if str(d.date())>=lo and str(d.date())<hi]); print(name,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(len(z))) if len(z)>1 else None)
fac.index=fac.index.strftime('%Y-%m-%d'); fac.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20270325_residual_5d_reversal_signal.csv',index=False)
