import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}
px=pd.DataFrame(D).sort_index(); fs={}
for s in U:
 p=px[s].dropna(); rr=p.pct_change(); m=p.pct_change(20); dd=rr.where(rr<0).rolling(20,min_periods=15).std(); fs[s]=(m/dd.replace(0,np.nan))
f=pd.DataFrame(fs); print('shape',px.shape,'range',px.index.min(),px.index.max(),'coverage',f.notna().mean().mean())
for h in [1,5,10]:
 fr=px.pct_change(h).shift(-h); a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'meanIC',np.nanmean(a),'std',np.nanstd(a,ddof=1),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'nmean',np.mean(ns))
print('period',px.index.min(),px.index.max())
print('corr_mom',f.stack().corr(px.pct_change(20).stack()))
