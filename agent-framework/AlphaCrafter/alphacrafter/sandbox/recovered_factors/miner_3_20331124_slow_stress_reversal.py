import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+sym+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close
px=pd.DataFrame({a:load(a) for a in assets}); v=load('VIX',1); d=load('DXY',1)
idx=px.index.intersection(v.index).intersection(d.index); px=px.reindex(idx); v=v.reindex(idx); d=d.reindex(idx)
# Slow stress-conditioned reversal: 20d asset return reversal, weighted by persistent 20d VIX elevation vs 120d baseline. Lag one day.
vr=v.pct_change(20); vz=(vr-vr.rolling(120).mean())/(vr.rolling(120).std()+1e-8)
sig=(-px.pct_change(20)).mul((vz.clip(-3,3)),axis=0).shift(1)
print('through',idx[-1].date(),'dates',len(idx),'assets',len(assets))
for h in [1,5,10,20]:
 f=px.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in idx:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),np.mean(z>0),len(z),np.mean(ns)))
# regime and conditional stress
f=px.pct_change(10).shift(-10); rows=[]
for dt in idx:
 ok=sig.loc[dt].notna()&f.loc[dt].notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic))
for lo,hi in [(2020,2023),(2024,2027),(2028,2030),(2031,2033)]:
 a=np.array([x for dt,x in rows if lo<=dt.year<=hi]);print('REG',lo,hi,len(a),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12))
r=sig.rank(axis=1,pct=True); print('coverage',sig.notna().mean().mean(),'turn10',np.nanmean((r-r.shift(10)).abs().mean(axis=1)))
