import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except Exception: pass
parts=[]
for s in U:
 x=get(s)
 if x is not None:
  x=x.set_index('date').sort_index(); r=x.close.pct_change()
  parts.append(pd.DataFrame({s+'_r':r,s+'_r20':x.close.pct_change(20),s+'_down':r.clip(upper=0).rolling(40).std()}))
d=pd.concat(parts,axis=1).sort_index()
r20=d[[s+'_r20' for s in U]].rename(columns=lambda z:z[:-4]); down=d[[s+'_down' for s in U]].rename(columns=lambda z:z[:-5])
breadth=(r20>0).mean(axis=1); gate=(breadth-.5).abs()+.5
f=(-r20/(down*np.sqrt(20)+1e-8)).mul(gate,axis=0).rolling(3,min_periods=3).mean().shift(1)
rets=d[[s+'_r' for s in U]].rename(columns=lambda z:z[:-2])
h=10; fr=(1+rets).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1
vals=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
q=pd.Series(vals); recent=q.tail(250)
print('assets',len(parts),'dates',len(q),'avg_n',np.mean(ns),'min_n',min(ns),'coverage',f.notna().sum(axis=1).mean()/15)
print('H10: IC=%.6f ICIR=%.6f hit=%.4f recent250=%.6f/%.6f' %(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
# regime split by cross-sectional dispersion of contemporaneous returns
cs=d[[s+'_r' for s in U]].rename(columns=lambda z:z[:-2]); disp=cs.std(axis=1)
for name,mask in [('low_disp',disp<=disp.quantile(.5)),('high_disp',disp>disp.quantile(.5))]:
 ix=[i for i,dt in enumerate(f.index) if mask.get(dt,False)]
 # align via dates recomputation
 v=[]
 for dt in f.index[mask.reindex(f.index,fill_value=False)]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(v); print(name,'dates',len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
