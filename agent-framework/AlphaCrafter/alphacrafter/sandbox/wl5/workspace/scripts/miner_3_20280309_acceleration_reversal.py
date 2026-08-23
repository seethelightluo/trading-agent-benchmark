import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
r20=p.pct_change(20); r60=p.pct_change(60)
# Acceleration reversal: fade deviation of recent 20d return from its 60d-per-20d baseline;
# normalize by 20d realized volatility and demean cross-sectionally.
acc=(r20-r60/3)
vol=r.rolling(20).std()*np.sqrt(20)
f=-(acc/(vol+1e-12))
f=f.sub(f.median(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean()))
df=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for label,z in [('all',df),('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 ic=z.ic
 print(label,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6) if len(ic)>1 else np.nan,'hit',round((ic>0).mean(),4))
print('coverage',round(df.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',df.date.min(),df.date.max())
out=f.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_3_20280309_acceleration_reversal_signal.csv',index=False)
