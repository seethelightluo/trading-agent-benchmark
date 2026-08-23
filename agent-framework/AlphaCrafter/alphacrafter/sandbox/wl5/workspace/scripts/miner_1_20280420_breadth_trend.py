import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
ret=p.pct_change(40); vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
above=(p>p.rolling(20,min_periods=15).mean()).sum(axis=1)/p.notna().sum(axis=1)
# In broad positive breadth follow risk-adjusted trend; otherwise use risk-adjusted reversal.
trend=ret/(vol+1e-12); reversal=-ret/(vol+1e-12)
f=trend.where(above>=0.5,reversal)
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1: rows.append((p.index[i],x[ok].corr(y[ok]),ok.sum()))
df=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
for label,z in [('all',df),('2020_23',df[(df.date>='2020-01-01')&(df.date<'2024-01-01')]),('2024_25',df[(df.date>='2024-01-01')&(df.date<'2026-01-01')]),('2026_27',df[(df.date>='2026-01-01')&(df.date<'2028-01-01')]),('2028',df[df.date>='2028-01-01'])]:
 q=z.ic; print(label,'dates',len(z),'avg_names',round(z.n.mean(),2) if len(z) else 0,'IC',round(q.mean(),6) if len(q) else np.nan,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan,'hit',round((q>0).mean(),4) if len(q) else np.nan)
print('dimensions',len(p),p.shape[1],'coverage',round(df.n.mean()/15,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',df.date.min(),df.date.max())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20280420_breadth_trend_signal.csv',index=False)
