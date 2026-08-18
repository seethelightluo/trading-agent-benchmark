import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date')['close']
P=pd.DataFrame(px).sort_index().ffill(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v.date); v=v.set_index('date')['close'].reindex(P.index).ffill()
rv=P.pct_change(5); vz=(v-v.rolling(60).mean())/v.rolling(60).std(); f=-rv*(1+0.75*vz.clip(lower=0,upper=2)); fr=P.shift(-10)/P-1
rows=[]; dates=[]; nobs=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt); nobs.append(ok.sum())
ic=np.array(rows,float); mean=float(np.nanmean(ic)); sd=float(np.nanstd(ic,ddof=1)); icir=mean/sd*np.sqrt(252) if sd else np.nan; turn=f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean()
print({'factor':'vix_conditioned_reversal_5d','dates':len(ic),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_instruments':float(np.mean(nobs)),'coverage':float(np.mean(nobs)/15),'ic':mean,'icir':icir,'hit_ratio':float(np.mean(ic>0)),'turnover':float(turn),'decay':'10-day forward horizon'})
for label,mask in [('pre2025',np.array([d.year<2025 for d in dates])),('2025plus',np.array([d.year>=2025 for d in dates]))]:
 x=ic[mask]; print(label,len(x),float(np.mean(x)) if len(x) else None,float(np.mean(x)/np.std(x,ddof=1)*np.sqrt(252)) if len(x)>1 else None)
