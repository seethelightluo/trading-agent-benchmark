import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-13')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 P[s]=d.close.astype(float)
p=pd.DataFrame(P).sort_index(); r=np.log(p/p.shift(1)); v=r.rolling(20,min_periods=15).std(); shock=v/v.rolling(60,min_periods=40).mean()
# short-term reversal amplified when volatility is elevated, smooth bounded weight
fac=-r.rolling(5,min_periods=5).sum()*(shock.clip(0.5,2.5))
fwd=p.shift(-5)/p-1
ics=[]; ns=[]; prev=None; turns=[]
for dt in p.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(ic);ns.append(len(z)); q=z.iloc[:,0].rank(pct=True)
  if prev is not None: turns.append(np.mean(abs(q-prev)))
  prev=q
x=np.array(ics); print({'factor':'volshock_reversal_5d','horizon':5,'cutoff':str(cut.date()),'dates':len(x),'avg_instruments':round(np.mean(ns),2),'coverage':round(np.mean(ns)/15,4),'IC':round(np.mean(x),6),'ICIR':round(np.mean(x)/np.std(x,ddof=1),6),'hit_ratio':round(np.mean(x>0),4),'turnover_rank':round(np.mean(turns),4)})
for l,a in [('early',x[:len(x)//2]),('late',x[len(x)//2:])]: print(l,round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),6),len(a))
