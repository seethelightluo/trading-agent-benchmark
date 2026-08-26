import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
root='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv(os.path.join(root,a+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[a]=d.close.replace(0,np.nan)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
raw=ret.rolling(20,min_periods=16).sum()/(ret.rolling(20,min_periods=16).std()*np.sqrt(252)); factor=(-raw).shift(1)
results={}
for h in [5,10,20,40,60]:
 fwd=prices.shift(-h)/prices-1; ics=[]; ns=[]
 for dt in factor.index:
  x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   v=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(v): ics.append(v); ns.append(ok.sum())
 z=np.asarray(ics); ic=z.mean(); icir=ic/z.std(ddof=1)
 results[h]=(len(z),np.mean(ns),np.mean(ns)/15,ic,icir,(z>0).mean())
 print(h, 'dates',len(z),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(ic,6),'ICIR',round(icir,6),'hit',round((z>0).mean(),4))
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20310320_volscaled_reversal_signal.csv',index=False)
print('turnover_proxy',float(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),'signal_rows',len(out))
