import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Defensive macro-conditioned short reversal: use only information available at date t.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# high-volatility state is VIX above its trailing 60-day median; no future data
state=(vix>vix.rolling(60,min_periods=30).median()).astype(float)
rev=-(r.rolling(5).sum())
# In stressed state, favor short reversal; in calm state shrink toward neutral to avoid noise
rank=rev.rank(axis=1,pct=True)
f=rank.sub(0.5).mul(0.5+0.5*state,axis=0)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean(),state.iloc[i]))
df=pd.DataFrame(rows,columns=['date','ic','coverage','stress']).dropna()
for lab,z in [('all',df),('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 print(lab,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(df.coverage.mean(),4),'turnover',round(f.diff().abs().mean().mean(),6),'stress_share',round(df.stress.mean(),4),'range',df.date.min(),df.date.max())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280504_macro_stress_reversal_signal.csv',index=False)
