import pandas as pd,numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date').close
c=pd.DataFrame(px).sort_index().ffill(); r=c.pct_change(); ret=c/c.shift(60)-1; vol=r.rolling(60,min_periods=40).std()*np.sqrt(60)
confirm=(c/c.shift(30)-1>0).astype(float)
f=(-(ret/(vol+1e-12))*confirm).rank(axis=1,pct=True).where(ret.notna())
y=c.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
print('cutoff',df.date.max().date(),'dates',len(df),'avgN',round(df.n.mean(),3),'IC',round(df.ic.mean(),6),'ICIR',round(df.ic.mean()/df.ic.std(ddof=1),6),'hit',round((df.ic>0).mean(),4))
for lab,z in [('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]: print(lab,'dates',len(z),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20280323_slow_reversal_confirm30d_signal.csv',index=False)
