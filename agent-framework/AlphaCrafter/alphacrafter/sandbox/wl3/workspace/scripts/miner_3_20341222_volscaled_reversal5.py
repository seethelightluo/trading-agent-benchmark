import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data

acct=get_account_dict(); syms=acct.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Factor: short-horizon reversal normalized by trailing volatility, lagged at each date.
f=(-P.pct_change(5)/(R.rolling(20).std()*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)
ics=[]; turnovers=[]; nvalid=[]
for i in range(25,len(P)-10):
 date=P.index[i]; names=f.columns[f.iloc[i].notna() & P.iloc[i+10].notna() & P.iloc[i].notna()]
 if len(names)<8: continue
 a=f.loc[date,names]; y=P.loc[P.index[i+10],names]/P.loc[date,names]-1
 ic=a.corr(y,method='spearman'); ics.append((date,ic)); nvalid.append(len(names))
 if i>25:
  prev=f.iloc[i-1][names].rank(pct=True); cur=a.rank(pct=True)
  turnovers.append(np.mean(abs(cur-prev)))
ser=pd.Series(dict(ics)).dropna();
for label,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]:
 print(label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'hit',round((z>0).mean(),4))
print('instruments',len(syms),'avg_valid',round(np.mean(nvalid),3),'coverage',round(np.mean(nvalid)/len(syms),4),'turnover',round(np.mean(turnovers),4),'period',P.index[0],P.index[-1])
# regime splits
for j,z in enumerate(np.array_split(ser,4),1): print('quartile',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
# artifact
out=pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}); out.to_csv('scripts/miner_3_20341222_volscaled_reversal5_signal.csv',index=False)
