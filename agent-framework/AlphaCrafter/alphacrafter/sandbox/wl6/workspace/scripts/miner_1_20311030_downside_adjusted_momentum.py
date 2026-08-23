import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); fs[s]=d.close.astype(float)
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-10-15']; r=p.pct_change()
# Positive momentum rewarded, but scale by downside risk; avoids lookahead and is interpretable.
ret10=p.pct_change(10)
down=r.where(r<0).rolling(30,min_periods=10).std()
sig=ret10/down.replace(0,np.nan)
print('candidate=downside_adjusted_momentum; dates=%d instruments=%d last=%s'%(len(p),len(U),p.index.max().date()),flush=True)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a))
 z=pd.Series(z)
 print('h=%d dates=%d avg_n=%.2f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()),flush=True)
f=p.shift(-10)/p-1; zz=[]; tt=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(q): tt.append(dt);zz.append(q)
z=pd.Series(zz,index=pd.DatetimeIndex(tt)); print('regimes10',z.groupby(z.index.year).agg(['mean','count']).round(6).to_dict('index'),flush=True)
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),flush=True)
