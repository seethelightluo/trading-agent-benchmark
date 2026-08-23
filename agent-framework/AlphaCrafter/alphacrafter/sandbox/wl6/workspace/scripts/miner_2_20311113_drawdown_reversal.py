import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 fs[s]=d.close.astype(float)
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-10-15']; r=p.pct_change()
# Contrarian signal: recent loss, amplified when price is deeply below its 60d peak,
# but normalized by realized volatility to avoid simply selecting crypto.
ret20=p.pct_change(20); dd=(p/p.rolling(60).max()-1.0); vol20=r.rolling(20).std()
sig=(-ret20)*(1+(-dd).clip(0,0.40))*vol20.pow(-1).replace([np.inf,-np.inf],np.nan)
print('candidate=drawdown_reversal; dates=%d instruments=%d last=%s'%(len(p),len(U),p.index.max().date()),flush=True)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a))
 z=pd.Series(z)
 print('h=%d dates=%d avg_n=%.2f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()),flush=True)
f=p.shift(-20)/p-1; zz=[];tt=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(q): tt.append(dt);zz.append(q)
z=pd.Series(zz,index=pd.DatetimeIndex(tt)); print('regimes20',z.groupby(z.index.year).agg(['mean','count']).round(6).to_dict('index'),flush=True)
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),flush=True)
