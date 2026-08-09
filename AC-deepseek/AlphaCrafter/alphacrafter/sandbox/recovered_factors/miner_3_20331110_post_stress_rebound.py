import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+sym+'.csv'
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); return d.set_index('date').sort_index()
prices={a:load(a).close for a in assets}; px=pd.DataFrame(prices)
vix=load('VIX',True).close; dxy=load('DXY',True).close
idx=px.index.intersection(vix.index).intersection(dxy.index); px=px.reindex(idx); vix=vix.reindex(idx); dxy=dxy.reindex(idx)
# continuous post-stress rebound: prior 5d asset loss, amplified by contemporaneous VIX shock; all lagged one day
ar=px.pct_change(5); vz=(vix.pct_change(5)-vix.pct_change(5).rolling(60).mean())/(vix.pct_change(5).rolling(60).std()+1e-8)
sig=(-ar*vz).shift(1)
# neutralize extreme macro direction? report
print('through',idx[-1].date(),'dates',len(idx),'assets',len(assets))
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[]; dates=[]
 for dt in idx:
  x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt)
 z=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),np.mean(z>0),len(z),np.mean([((sig.loc[d].notna())&(fwd.loc[d].notna())).sum() for d in dates])))
# turnover rank proxy 10d
r=sig.rank(axis=1,pct=True); print('coverage',sig.notna().mean().mean(),'turn10',np.nanmean((r-r.shift(10)).abs().mean(axis=1)))
# regime h10
fwd=px.pct_change(10).shift(-10); zlist=[]
for dt in idx:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:zlist.append((dt,spearmanr(x[ok],y[ok]).statistic))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 a=np.array([v for d,v in zlist if lo<=str(d.year)<=hi]); print('REG',lo,hi,len(a),np.nanmean(a) if len(a) else np.nan,np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12) if len(a)>1 else np.nan)
