import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index().close
px=pd.DataFrame({a:load(a) for a in assets}).sort_index()
# Volatility-scaled medium-term momentum: 60d return divided by 20d realized volatility, lagged one day.
r=px.pct_change(); sig=(px.pct_change(60)/(r.rolling(20).std()*np.sqrt(20)+1e-8)).shift(1)
print('through',px.index[-1].date(),'dates',len(px),'assets',len(assets))
for h in [1,5,10,20]:
 f=px.pct_change(h).shift(-h);z=[];ns=[]
 for dt in px.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('H',h,'dates',len(z),'meanN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(z),np.mean(z)/(np.std(z,ddof=1)+1e-12),np.mean(z>0)))
for lo,hi in [(2020,2023),(2024,2027),(2028,2030),(2031,2033)]:
 f=px.pct_change(10).shift(-10);z=[]
 for dt in px.index:
  if not(lo<=dt.year<=hi):continue
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic)
 z=np.array(z);print('REG',lo,hi,'dates',len(z),'IC %.6f ICIR %.6f'%(np.mean(z),np.mean(z)/(np.std(z,ddof=1)+1e-12)))
rk=sig.rank(axis=1,pct=True);print('coverage %.4f turn10 %.4f'%(sig.notna().mean().mean(),np.nanmean((rk-rk.shift(10)).abs().mean(axis=1))))
