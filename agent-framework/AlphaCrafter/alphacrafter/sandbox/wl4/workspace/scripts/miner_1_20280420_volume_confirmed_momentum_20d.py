import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); closes={}; vols={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),usecols=['date','close','volume']); d['date']=pd.to_datetime(d.date); d=d.set_index('date'); closes[s]=d.close; vols[s]=d.volume
p=pd.DataFrame(closes).sort_index(); v=pd.DataFrame(vols).reindex(p.index)
r=p.pct_change()
# Lag all inputs one completed session: momentum confirmed by abnormal recent volume.
mom=p.shift(1)/p.shift(21)-1
vr=(v.rolling(5,min_periods=3).mean()/v.rolling(60,min_periods=20).mean()).shift(1)
f=(mom*np.log(vr.clip(lower=0.05))).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.median(axis=1),axis=0)
print('universe_dates=%d assets=%d' % (len(p),p.shape[1]))
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; counts=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic); counts.append(ok.sum())
 x=pd.Series(vals).dropna(); print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f' %(h,len(x),np.mean(counts),np.mean(counts)/15,x.mean(),x.mean()/x.std(),np.mean(x>0)))
 if len(x):
  for name,z in [('early',x.iloc[:len(x)//2]),('late',x.iloc[len(x)//2:])]: print(' %s dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(name,len(z),z.mean(),z.mean()/z.std(),np.mean(z>0)))
rank=f.rank(axis=1,pct=True); print('turnover=%.6f' % rank.diff().abs().mean(axis=1).dropna().mean())
