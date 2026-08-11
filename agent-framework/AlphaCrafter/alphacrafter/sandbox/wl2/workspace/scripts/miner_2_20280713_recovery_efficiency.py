import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
# recovery efficiency: distance recovered from 60d low, scaled by recent realized risk
close=pd.DataFrame({s:D[s].close for s in U}); ret=close.pct_change()
vol20=ret.rolling(20,min_periods=15).std()
raw=(close/close.rolling(60,min_periods=45).min()-1)/(vol20*np.sqrt(20))
f=raw.shift(1)
for h in [1,5,10]:
    fr=close.shift(-h)/close-1
    vals=[]; ns=[]; turns=[]
    prev=None
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
            ns.append(len(z))
            r=x.rank(pct=True); turns.append(np.nan if prev is None else (r-prev).abs().mean())
            prev=r
    a=np.array(vals); a=a[np.isfinite(a)]
    print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),3),'turn',round(np.nanmean(turns),4))
# regime split
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-07-12')]:
    sub=f.loc[lo:hi]; fr=(close.shift(-1)/close-1).loc[lo:hi]; a=[]
    for dt in sub.index:
      z=pd.concat([sub.loc[dt],fr.loc[dt]],axis=1).dropna()
      if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=np.array(a);print('REG',lo,hi,'dates',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4))
