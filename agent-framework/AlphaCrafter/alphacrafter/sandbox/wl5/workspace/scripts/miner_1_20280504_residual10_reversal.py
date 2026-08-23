import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std(); cs=r.mean(axis=1); res=r.sub(cs,axis=0)
# Residual 10-day mean-reversion, volatility normalized; signal is lagged at decision date.
raw=-(res.rolling(10).sum()/(vol*np.sqrt(10)+1e-12)); f=raw.rank(axis=1,pct=True).sub(.5)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean()))
df=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
print('dates',len(df),'avg_names',round(df.coverage.mean()*len(U),2),'IC',round(df.ic.mean(),6),'ICIR',round(df.ic.mean()/df.ic.std(ddof=1),6),'hit',round((df.ic>0).mean(),4),'turnover',round(f.diff().abs().mean().mean(),6))
for lab,q in [('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]: print(lab,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20280504_residual10_reversal_signal.csv',index=False)
