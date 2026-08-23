import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill()
r=p.pct_change()
# Candidate: short momentum normalized by own volatility, demeaned by cross-sectional median (breadth-aware)
mom=p.pct_change(15)
vol=r.rolling(20).std()*np.sqrt(20)
raw=mom/(vol+1e-12)
csmed=raw.median(axis=1)
# activate only when broad market median 15d return positive: trend continuation, else inverse signal
breadth=(mom>0).mean(axis=1)
f=raw.sub(raw.median(axis=1),axis=0).mul(pd.Series(np.where(breadth.values>=0.5,1,-1),index=raw.index),axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-10):
 dt=p.index[i]
 if pd.isna(f.iloc[i]).sum()>7: continue
 fr=p.iloc[i+10]/p.iloc[i]-1
 x=f.iloc[i]; ok=x.notna()&fr.notna()
 if ok.sum()>=8:
  ic=x[ok].corr(fr[ok]); rows.append((dt,ic,ok.mean()))
df=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for label, z in [('all',df),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 ic=z.ic
 print(label,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6) if len(ic)>1 else np.nan,'hit',round((ic>0).mean(),4))
print('coverage',round(df.coverage.mean(),4))
# rank turnover proxy
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean().mean(),6))
print('range',df.date.min(),df.date.max())
# emit artifact values for provenance
out=f.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_3_20280224_breadth_regime_momentum_signal.csv',index=False)
