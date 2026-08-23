import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={};V={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d): z=d.set_index('date'); C[s]=z.close; V[s]=z.volume
p=pd.DataFrame(C).sort_index().ffill();v=pd.DataFrame(V).reindex(p.index).ffill();r=p.pct_change()
# Short momentum confirmed by sustained volume expansion, volatility scaled.
volshock=(v.rolling(3).mean()/(v.rolling(40).mean()+1e-12)-1).clip(-1,3)
f=(r.rolling(5).sum()/(r.rolling(15).std()*np.sqrt(15)+1e-6))*(1+0.3*volshock)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01']),('recent',d[d.date>='2027-08-24'])]:
 if len(z): print(n,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(d.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',d.date.min(),d.date.max(),'instruments',len(C))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20280824_short_volume_momentum_signal.csv',index=False)
