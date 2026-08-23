import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# medium-horizon return scaled by lagged downside risk; rewards persistent positive trend and penalizes asymmetric drawdown risk
r20=np.log(p/p.shift(20)); r60=np.log(p/p.shift(60))
down=(r.where(r<0,0.0)**2).rolling(40).mean().pow(.5).shift(1)
vol=r.rolling(40).std().shift(1)
f=(r20/(down+0.35*vol+1e-12) * (1+0.25*np.sign(r20)*np.sign(r60))).clip(-10,10)
rows=[]
for d in f.index:
 fut=np.log(p.shift(-10)/p).loc[d]; a=f.loc[d]; ok=a.notna()&fut.notna()
 if ok.sum()>=8: rows.append((d,a[ok].rank().corr(fut[ok].rank()),ok.sum()))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor=downside_adjusted_momentum dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for k in [180,360]: print('recent',k,q.tail(k).ic.mean(),q.tail(k).ic.mean()/q.tail(k).ic.std(ddof=1))
for h in [5,20]:
 z=[]
 for d in f.index:
  fut=np.log(p.shift(-h)/p).loc[d]; a=f.loc[d]; ok=a.notna()&fut.notna()
  if ok.sum()>=8:z.append(a[ok].rank().corr(fut[ok].rank()))
 print('decay',h,np.nanmean(z))
# save deterministic signal artifact
f.rename_axis('date').to_csv('scripts/miner_1_20300502_downside_momentum_signal.csv')
q.to_csv('scripts/miner_1_20300502_downside_momentum_ic.csv')
