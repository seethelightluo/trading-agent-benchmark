import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for a in assets}
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); med=r.median(axis=1); ex=r.sub(med,axis=0)
# Candidate: medium-horizon residual momentum scaled only by downside volatility.
# Positive residual trend is preferred, but assets with unstable downside moves are penalized.
F=[]
for i,dt in enumerate(px.index):
 if i<65: continue
 trend=ex.iloc[i-20:i].sum()
 dn=ex.iloc[i-60:i].where(ex.iloc[i-60:i]<0).std().replace(0,np.nan)
 F.append((trend/(dn+1e-6)).rename(dt))
F=pd.DataFrame(F)
def calc(h, idx=F.index):
 ic=[]; ns=[]
 for dt in idx:
  i=px.index.get_loc(dt)
  if i+h>=len(px): continue
  z=pd.concat([F.loc[dt],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(ic); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
for h in [1,5,10,20]: print('H',h,'dates mean_n IC ICIR hit', [round(v,6) if isinstance(v,float) else v for v in calc(h)])
rank=F.rank(axis=1,pct=True); print('turnover10',((rank-rank.shift(10)).abs().mean(axis=1)).dropna().mean(),'coverage',F.notna().mean().mean(),'dates',len(F),'assets',len(assets))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 q=calc(10,F.loc[lo:hi].index);print('REG',lo,hi,'dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
q=calc(10,F.index[-120:]);print('RECENT120','dates',q[0],'IC',round(q[2],6),'ICIR',round(q[3],6))
F.to_csv('/tmp/downside_mom.csv')
