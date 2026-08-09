import numpy as np, pandas as pd, glob, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def dat(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): return None
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); return d.sort_values('date')
allr={}; allf={}
for s in U:
 d=dat(s)
 if d is None: continue
 r=pd.to_numeric(d.close,errors='coerce').pct_change(); mom=d.close.pct_change(15); vol=r.rolling(20).std()*np.sqrt(20)
 allf[s]=pd.Series((mom/vol.replace(0,np.nan)).values,index=d.date)
 allr[s]=pd.Series(pd.to_numeric(d.close).pct_change().shift(-1).values,index=d.date)
x=pd.DataFrame(allf).sort_index(); y=pd.DataFrame(allr).reindex(x.index)
ics=[]; turnovers=[]; counts=[]; dates=[]; prev=None
for dt in x.index:
 a=x.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ic=a[ok].rank().corr(b[ok].rank()); ics.append(ic);counts.append(ok.sum());dates.append(dt)
  cur=(a.rank()/(ok.sum()+1)).fillna(0)
  if prev is not None: turnovers.append((cur-prev).abs().sum()/2)
  prev=cur
ics=np.array(ics); dates=pd.DatetimeIndex(dates)
print('dates',len(ics),'avg_n',np.mean(counts),'coverage',np.mean(np.array(counts)/15),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover',np.mean(turnovers))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15'),('2026-07-16','2027-02-25')]:
 mask=(dates>=pd.Timestamp(a))&(dates<=pd.Timestamp(b)); print(a,b,mask.sum(),np.mean(ics[mask]) if mask.sum() else np.nan)
for h in [5,10]:
 yy={}
 for s in U:
  d=dat(s)
  if d is not None: yy[s]=pd.Series(pd.to_numeric(d.close).pct_change(h).shift(-h).values,index=d.date)
 yh=pd.DataFrame(yy).reindex(x.index); zz=[]
 for dt in x.index:
  ok=x.loc[dt].notna()&yh.loc[dt].notna()
  if ok.sum()>=8: zz.append(x.loc[dt,ok].rank().corr(yh.loc[dt,ok].rank()))
 zz=np.array(zz); print('h',h,'dates',len(zz),'IC',np.nanmean(zz),'ICIR',np.nanmean(zz)/np.nanstd(zz,ddof=1))
