import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-25')
D={}; C={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=cut]; D[s]=d; C[s]=d.close
close=pd.DataFrame(C).sort_index(); op=pd.DataFrame({s:D[s].open for s in syms}).reindex(close.index); hi=pd.DataFrame({s:D[s].high for s in syms}).reindex(close.index); lo=pd.DataFrame({s:D[s].low for s in syms}).reindex(close.index)
# prior session intraday exhaustion: negative close location (close near high is fade) normalized by range
rng=(hi-lo).replace(0,np.nan); f=-(close-op)/rng; f=f.shift(1)
for h in [1,5,10]:
 y=close.shift(-h)/close-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.median(ns))
print('coverage',f.notna().mean().mean(),'turn',f.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270325_intraday_exhaustion_signal.csv',index=False)
