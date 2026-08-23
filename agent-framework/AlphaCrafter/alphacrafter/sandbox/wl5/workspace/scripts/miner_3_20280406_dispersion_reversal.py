import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
base=r.rolling(5).sum()/(r.rolling(20).std()*np.sqrt(20)+1e-12)
disp=r.rolling(5).std().mean(axis=1)
# percentile of lagged dispersion over trailing 20 observations; current value is known at t
rank=disp.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)
mult=1.0+0.75*rank.clip(0,1)
f=(-base.mul(mult,axis=0))
f=f.sub(f.median(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean()))
df=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for lab,z in [('all',df),('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 print(lab,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(df.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',df.date.min(),df.date.max())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280406_dispersion_reversal_signal.csv',index=False)
