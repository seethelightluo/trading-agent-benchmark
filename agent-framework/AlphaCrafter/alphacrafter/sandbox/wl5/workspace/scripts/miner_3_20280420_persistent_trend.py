import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()+1e-12
m20=r.rolling(20).sum()/vol/np.sqrt(20); m5=r.rolling(5).sum()/vol/np.sqrt(5)
# persistence trend: medium risk-adjusted momentum, gated by agreement with short horizon
agree=np.sign(m20)*np.sign(m5)
f=m20.rank(axis=1,pct=True)*0.7+m5.rank(axis=1,pct=True)*0.3
f=f.where(agree>0, 0.5).sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean()))
df=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for lab,z in [('all',df),('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 print(lab,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(df.coverage.mean(),4),'turnover',round(f.diff().abs().mean().mean(),6),'range',df.date.min(),df.date.max())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280420_persistent_trend_signal.csv',index=False)
