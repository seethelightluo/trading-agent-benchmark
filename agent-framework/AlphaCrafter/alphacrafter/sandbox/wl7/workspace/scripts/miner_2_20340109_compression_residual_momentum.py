import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=2200) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=np.log(px).diff()
# Candidate: residual medium momentum, strengthened by volatility compression and volume unavailable
mom=r.rolling(20).sum(); vol=r.rolling(20).std(); longvol=r.rolling(60).std()
res=mom.sub(mom.mean(axis=1),axis=0)
# compression ratio, clipped; low vol gets positive multiplier, avoid extreme
comp=(longvol/(vol+1e-12)).clip(0.5,2.0)
f=(res/(vol+1e-12))*comp
# lag so no lookahead
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
  fr=np.log(px.shift(-h)/px)
  vals=[]; dates=[]; ns=[]
  for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]
    ok=a.notna()&b.notna()
    if ok.sum()>=8:
      vals.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt); ns.append(ok.sum())
  z=pd.Series(vals,index=dates).dropna()
  print('H',h,'dates',len(z),'avgN',round(float(np.mean(ns)),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
# recent thirds for H10
h=10; fr=np.log(px.shift(-h)/px); vals=[]; dates=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8:
  q=f.loc[dt][ok].corr(fr.loc[dt][ok],method='spearman')
  if pd.notna(q): vals.append(q); dates.append(dt)
z=pd.Series(vals,index=dates)
print('thirds', [round(x.mean(),6) for x in np.array_split(z,3)])
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
# artifact signal
out=f.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_2_20340109_compression_residual_momentum_signal.csv',index=False)
print('range',out.date.min(),out.date.max())
