import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)==0:return None
 x=d.copy();x['date']=pd.to_datetime(x['date']).dt.normalize();return x.set_index('date')['close']
px={s:close(s) for s in U};px={s:x for s,x in px.items() if x is not None}
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change()
ret=r.rolling(10).sum();disp=r.std(axis=1).rolling(20).mean(); z=(disp-disp.rolling(120).mean())/(disp.rolling(120).std()+1e-9); mult=np.clip(1+0.5*z,-1,3)
f=-ret.mul(mult,axis=0)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z0 in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01'])]:
 if len(z0):print(n,'dates',len(z0),'avg_names',round(z0.coverage.mean()*len(U),2),'IC',round(z0.ic.mean(),6),'ICIR',round(z0.ic.mean()/z0.ic.std(ddof=1),6),'hit',round((z0.ic>0).mean(),4))
print('coverage',round(d.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',d.date.min(),d.date.max(),'instruments',len(px))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20280727_dispersion_reversal_signal.csv',index=False)
