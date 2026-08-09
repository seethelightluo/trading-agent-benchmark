import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
px=pd.concat(D,axis=1).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
disp=r.sub(r.median(axis=1),axis=0).abs().mean(axis=1); hi=disp.rolling(120,min_periods=60).rank(pct=True)>=.7
f=-(px.pct_change(5)/vol).where(hi.to_numpy()[:,None]); fr={h:px.pct_change(h).shift(-h) for h in [1,5,10,20]}
for h,y in fr.items():
  q=[]; ns=[]
  for dt in px.index:
    z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
    if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=np.asarray(q); print(f'candidate highdisp_rev5 H{h}: dates={len(q)} meanN={np.mean(ns):.2f} IC={np.nanmean(q):.6f} ICIR={np.nanmean(q)/np.nanstd(q,ddof=1):.6f} hit={np.mean(q>0):.4f}')
# Correctly aligned pooled correlation against broad admitted-style reconstructed signals.
sigs={'trend20':px.pct_change(20)/vol,'rev5':-px.pct_change(5)/r.rolling(5,min_periods=4).std(),'invvol':-vol,'trend60':px.pct_change(60),'invkurt':-r.rolling(40,min_periods=30).kurt(),'invskew':-r.rolling(40,min_periods=30).skew()}
cs=[]
for name,s in sigs.items():
 z=pd.concat([f.stack().rename('candidate'),s.stack().rename(name)],axis=1).dropna()
 rho=spearmanr(z['candidate'].to_numpy(),z[name].to_numpy()).statistic
 cs.append((name,abs(rho),rho,len(z)))
print('aligned_proxy_correlations',sorted(cs,key=lambda x:-x[1]))
print('candidate_coverage',int(f.notna().sum().sum()),int(f.size),f.notna().stack().mean())
print('active_dates',int(hi.sum()),'universe',len(assets))
# regime split
for lo,hi_dt in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2031-10-29')]:
 y=fr[5]; mask=(px.index>=lo)&(px.index<=hi_dt); q=[]
 for dt in px.index[mask]:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi_dt,'dates',len(q),'IC',np.nanmean(q) if q else np.nan,'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1) if len(q)>1 else np.nan)
