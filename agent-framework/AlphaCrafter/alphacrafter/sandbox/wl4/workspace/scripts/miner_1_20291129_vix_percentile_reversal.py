import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); idx=Path('../persistent/index_data')
px={a:pd.read_csv(base/f'{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
vix=pd.read_csv(idx/'VIX.csv',parse_dates=['date']).set_index('date')['close'].rename('vix')
panel=pd.concat(px,axis=1).sort_index().ffill()
ret=panel.pct_change()
# Observation-only VIX, aligned without looking ahead; factor itself is lagged one day.
vr=vix.reindex(panel.index).ffill()
q=vr.rolling(120,min_periods=60).quantile(.70)
vol=ret.rolling(20,min_periods=15).std()
r20=panel.pct_change(20)
raw=(-r20/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
# binary percentile stress gates, with the same reversal score in stressed regimes
variants={'vix70_binary':raw.where(vr>q,0.0),'vix80_binary':raw.where(vr>vr.rolling(120,min_periods=60).quantile(.80),0.0),'vix70_soft':raw*(1+0.75*(vr>q).astype(float))}
for name,f in variants.items():
  f=f.shift(1)
  rows=[]
  for h in [1,5,10,20]:
    fw=panel.pct_change(h).shift(-h)
    ics=[]; ns=[]; turns=[]
    prev=None
    for d in panel.index:
      x=f.loc[d]; y=fw.loc[d]
      z=pd.concat([x,y],axis=1).dropna()
      if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
        ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
      if prev is not None:
        turns.append((f.loc[d].rank(pct=True)-prev).abs().mean())
      prev=f.loc[d].rank(pct=True)
    a=np.array(ics); ic=np.nanmean(a); icir=ic/np.nanstd(a,ddof=1)*np.sqrt(len(a)) if len(a)>1 else np.nan
    print(name,'h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(icir,6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(turns),6))
  # recent 250 at 10d
  fw=panel.pct_change(10).shift(-10); ics=[]
  for d in panel.index[-400:]:
    z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(ics[-250:]); print(name,'recent250_10d',len(a),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),6))
