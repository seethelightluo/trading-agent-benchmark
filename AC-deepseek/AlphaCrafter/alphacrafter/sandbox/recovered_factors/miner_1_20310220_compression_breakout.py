import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms};p=pd.DataFrame(p).sort_index();r=p.pct_change();d=p.index
# Volatility compression followed by directional breakout: recent 5d move scaled by
# prior 20d volatility, activated when prior volatility is in its lower 45% history.
vol=r.rolling(20).std(); q=vol.rolling(120).rank(pct=True)
f=(p.pct_change(5)/vol.shift(5)).where(q.shift(1)<.45)
# cross-sectional demean (avoids common market direction)
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; a=[];ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'N',np.mean(ns))
# regime H10
fr=p.shift(-10)/p-1; o=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:o.append((d[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
v=pd.DataFrame(o,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-10','2031-02')]:
 x=v.loc[a:b,'ic'];print('REG',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'active dates',f.notna().any(axis=1).sum(),'N',f.notna().sum(axis=1).mean(),'turn10',f.rank(axis=1).diff(10).abs().mean().mean())
