import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
syms=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Lagged residual short-term reversal: remove common cross-asset daily move,
# then favor 5d losers, scaled by trailing 20d idiosyncratic volatility.
common=R.mean(axis=1)
res=R.sub(common,axis=0)
vol=res.rolling(20).std()
f=(-res.rolling(5).sum()/vol).shift(1)
ics=[]; nv=[]; tv=[]
for i in range(25,len(P)-10):
 names=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(names)<8: continue
 a=f.iloc[i][names]; y=P.iloc[i+10][names]/P.iloc[i][names]-1
 ics.append((P.index[i],a.corr(y,method='spearman'))); nv.append(len(names))
 if i>25:
  old=f.iloc[i-1][names]
  tv.append(np.mean(abs(a.rank(pct=True)-old.rank(pct=True))))
ser=pd.Series(dict(ics)).dropna()
for label,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]: print(label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('instruments',len(syms),'avg_valid',round(np.mean(nv),3),'coverage',round(np.mean(nv)/len(syms),4),'turnover',round(np.mean(tv),4),'period',P.index[0],P.index[-1])
for j,z in enumerate(np.array_split(ser,4),1): print('quartile',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':ser.index,'factor_ic':ser.values}).to_csv('scripts/miner_1_20350119_residual_reversal5_signal.csv',index=False)
