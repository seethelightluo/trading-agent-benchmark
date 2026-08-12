import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
u=get_account_dict().get('watch_list',[])
if not u: u=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): px[s]=d.set_index('date')
cl=pd.DataFrame({s:d['close'] for s,d in px.items()}).sort_index(); ret=np.log(cl).diff()
# Range/volatility-conditioned 3-day reversal: emphasize abrupt moves with close-to-close shock,
# but penalize wide noisy daily ranges. Lag one completed day.
atr=(pd.DataFrame({s: (d['high']-d['low'])/d['close'] for s,d in px.items()}).reindex(cl.index)).rolling(20).mean()
rv=ret.rolling(60).std()*np.sqrt(3)
shock=-ret.rolling(3).sum()/rv
range_pen=(ret.abs().rolling(10).mean()/atr).clip(lower=.2,upper=5)
f=shock/range_pen
f=f.shift(1)
rows=[]
for i in range(len(cl)-10):
 t=cl.index[i]; fut=ret.iloc[i+1:i+11].sum()
 for h in [1,3,5,10]:
  if i+h>=len(cl): continue
  x=f.iloc[i]; y=ret.iloc[i+1:i+h+1].sum()
  z=pd.concat([x,y],axis=1).dropna();
  if len(z)>=8: rows.append((t,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('universe',len(u),'usable',len(px),'dates',len(cl),'observations',len(df))
for h in [1,3,5,10]:
 z=df[df.h==h]; m=z.ic.mean(); sd=z.ic.std(ddof=1); print(h,'dates',len(z),'avg_n',z.n.mean(),'IC',m,'ICIR',m/sd,'hit', (z.ic>0).mean())
print('regimes')
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 z=df[(df.h==1)&(df.date.astype(str)>=a)&(df.date.astype(str)<=b+'-12-31')]; print(a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
# coverage/turnover proxy
valid=f.notna().sum(axis=1)/len(u); ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print('coverage',valid.mean(),'turnover',turnover)
f.to_csv('scripts/miner_1_20290906_range_conditioned_reversal_signal.csv')
print('signal_artifact scripts/miner_1_20290906_range_conditioned_reversal_signal.csv')
