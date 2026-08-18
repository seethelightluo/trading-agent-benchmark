import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
acct=get_account_dict(); syms=acct.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Volatility-normalized 20-session trend, lagged one observation and tested at 10d.
f=(P.pct_change(20)/(R.rolling(20).std()*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)
ics=[]; nv=[]; tv=[]
for i in range(25,len(P)-10):
 names=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(names)<8: continue
 a=f.iloc[i][names]; y=P.iloc[i+10][names]/P.iloc[i][names]-1
 ics.append((P.index[i],a.corr(y,method='spearman'))); nv.append(len(names))
 if i>25:
  tv.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][names].rank(pct=True))))
ser=pd.Series(dict(ics)).dropna()
for label,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]: print(label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('instruments',len(syms),'avg_valid',round(np.mean(nv),3),'coverage',round(np.mean(nv)/len(syms),4),'turnover',round(np.mean(tv),4),'period',P.index[0],P.index[-1])
for j,z in enumerate(np.array_split(ser,4),1): print('quartile',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}).to_csv('scripts/miner_3_20350105_voltrend20_signal.csv',index=False)
