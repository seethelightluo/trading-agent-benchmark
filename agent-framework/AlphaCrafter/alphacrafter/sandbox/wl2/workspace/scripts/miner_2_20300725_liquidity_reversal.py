import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}; vol={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<100: d=get_index_daily_data(s,2600)
 if d is not None:
  d=d.set_index('date'); close[s]=d.close.astype(float); vol[s]=d.volume.astype(float)
P=pd.DataFrame(close).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change()
rows=[]; signals=[]
# Liquidity-weighted short-term reversal: reverse the 3-day residual return,
# scaled by 20d volatility and attenuated for illiquid assets via relative volume.
# Relative volume is trailing-only and cross-sectionally demeaned each date.
for t in range(80,len(P)-11):
 r=R.iloc[t-2:t+1].sum(); med=R.iloc[t-2:t+1].median(axis=1).sum(); vv=R.iloc[t-19:t+1].std()
 rv=(V.iloc[t]/V.iloc[t-20:t].median()).replace([np.inf,-np.inf],np.nan)
 liq=np.log(rv.clip(lower=0.25,upper=4.0)); liq=liq-liq.median()
 f=(-(r-med)/vv.replace(0,np.nan))*(1+0.25*liq)
 f=f.replace([np.inf,-np.inf],np.nan).dropna()
 signals.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index); q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(signals); S.to_csv('scripts/miner_2_20300725_liquidity_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'instruments',len(U),'available',len(close))
