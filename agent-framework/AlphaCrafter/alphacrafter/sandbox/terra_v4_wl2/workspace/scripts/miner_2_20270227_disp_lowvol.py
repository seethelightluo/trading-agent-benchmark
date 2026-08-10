import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A};P=pd.DataFrame(p);r=P.pct_change();
# dispersion regime, cross-sectional defensive low-vol score (negative realized vol), lagged activation
D=r.std(axis=1).rolling(20,min_periods=15).mean(); q=D.rolling(120,min_periods=60).quantile(.7); active=(D>q).shift(1)
f=(-r.rolling(20,min_periods=15).std()).where(active); f=f.sub(f.median(axis=1),axis=0)
rows=[];sig=[]
for dt in P.index:
 for a in A:sig.append((dt,a,f.loc[dt,a]))
 for h in [1,3,5,10]:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1;z=pd.concat([f.loc[dt].rename('f'),y.rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:rows.append((dt,h,spearmanr(z.f,z.y).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10]:
 x=d[d.h==h];print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2024-12'),('2025','2025-12'),('2026','2026-12'),('2027','2027-02-25')]:
  a=x.set_index('date').loc[lo:hi].ic;print(lo,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270227_disp_lowvol.csv',index=False);print('coverage',round(out.signal.notna().mean(),4),'active',active.sum())
