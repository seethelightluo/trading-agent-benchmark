import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']).dt.normalize(); x=x.set_index('date')['close'].astype(float)
        D[s]=x
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# Candidate: dispersion-conditioned residual 10d reversal. Cross-sectional average return is removed;
# signal is negative idiosyncratic 10d move, strengthened when cross-sectional dispersion is high.
ret10=p/p.shift(10)-1
csmean=ret10.mean(axis=1)
res=ret10.sub(csmean,axis=0)
disp=ret10.std(axis=1)
# normalize by own 20d vol, and use only lagged dispersion regime (continuous bounded multiplier)
vol=r.rolling(20).std()*np.sqrt(252)
base=-res/vol
mult=(disp/disp.rolling(60).median()).shift(1).clip(0.5,2.0)
f=base.mul(mult,axis=0)
fr=f.shift(1); fwd=p.shift(-10)/p-1
rows=[]; dates=[]
for dt in f.index:
    a=fr.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        rows.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt)
ic=pd.Series(rows,index=dates).dropna()
# rank turnover
ranks=f.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean()
print('candidate=dispersion_conditioned_residual_reversal_10d')
print('dates',len(ic),'instruments_avg',round(np.mean([((fr.loc[d].notna())&(fwd.loc[d].notna())).sum() for d in dates]),2),'period',ic.index.min().date(),ic.index.max().date())
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f coverage %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),turn,p.notna().mean().mean()))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028')]:
 q=ic.loc[a:b]
 print(a,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
print('recent',ic.tail(252).mean(),ic.tail(252).mean()/ic.tail(252).std())
# decay 5/10/20
for h in [5,10,20]:
 z=[]
 for dt in f.index:
  a=fr.loc[dt]; b=p.shift(-h).loc[dt]/p.loc[dt]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(a[ok].corr(b[ok],method='spearman'))
 z=pd.Series(z).dropna();print('decay',h,len(z),round(z.mean(),6),round(z.mean()/z.std(),6))
# artifact signals latest all dates, enough for audit
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20280907_dispersion_residual_reversal_signal.csv',index=False)
