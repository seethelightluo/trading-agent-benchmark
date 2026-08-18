import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 P[s]=x['close'].astype(float); V[s]=x['volume'].astype(float) if 'volume' in x else pd.Series(index=x.index,dtype=float)
p=pd.concat(P,axis=1).sort_index().loc[:'2028-12-13']; v=pd.concat(V,axis=1).reindex(p.index); r=p.pct_change()
rv20=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(252); rv60=r.rolling(60,min_periods=40).std().shift(1)*np.sqrt(252)
base=(p.shift(1)/p.shift(21)-1)/rv20-(p.shift(1)/p.shift(61)-1)/rv60
# Lagged volume confirmation: recent volume relative to its 60-session median, clipped to avoid domination.
vr=(v.shift(1).rolling(20,min_periods=10).mean()/v.shift(1).rolling(60,min_periods=30).median()).clip(0.5,2.0)
f=(base*vr).replace([np.inf,-np.inf],np.nan); f=f.sub(f.mean(axis=1),axis=0)
print('candidate=volume_confirmed_vol_adjusted_trend_acceleration cutoff=2028-12-13')
for h in [1,5,10,20]:
 a=[]; ns=[]; ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c);ns.append(len(z));ds.append(t)
 q=pd.Series(a,index=ds); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5),'valid_dates',f.notna().any(axis=1).sum(),'assets',len(U))
