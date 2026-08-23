import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-18')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=end]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# volatility acceleration: recent 20d vol relative to 60d vol, with small reversal overlay
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
raw=v20/v60
f=raw.rank(axis=1,pct=True)-0.20*r.rolling(5,min_periods=5).sum().rank(axis=1,pct=True)
rows=[]; sig=[]
for i in range(60,len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((p.index[i],ic,len(z)))
  sig.append((p.index[i],*x.reindex(assets).values))
ics=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ics),'avg_n',ics.n.mean(),'IC',ics.ic.mean(),'ICIR',ics.ic.mean()/ics.ic.std(ddof=1),'hit',(ics.ic>0).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-05-18')]:
 q=ics.loc[lo:hi].ic; print(lo,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# turnover rank ordering proxy mean absolute change in normalized cross-sectional factor
sf=pd.DataFrame([x[1:] for x in sig],index=[x[0] for x in sig],columns=assets)
turn=sf.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('coverage',ics.n.mean()/15,'turnover',turn)
out='scripts/miner_1_20280518_vol_accel_reversal_signal.csv'; sf.to_csv(out); print('artifact',out)
