import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,4000) for s in U}
for s in U:
 d=px[s]; d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date')['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]
for d in dates:
 vals=[]
 for s in U:
  h=px[s].loc[:d].pct_change().dropna()
  if len(h)<61: vals.append((s,np.nan)); continue
  x=h.iloc[-60:-1]
  up=x[x>0].mean(); down=(-x[x<0]).mean(); total=x.sum()
  # persistent upside participation relative to downside, neutralized cross-section
  vals.append((s,(total/(x.std()+1e-12)) + 0.8*(up/(down+1e-12)-1)))
 sig=pd.Series(dict(vals)); sig=sig-sig.median()
 for s in U:
  fut=px[s].loc[px[s].index>d]
  if pd.notna(sig[s]) and len(fut)>=10:
   for H in [1,3,5,10]: rows.append((d,s,float(sig[s]),H,float(fut.iloc[H-1]/px[s].loc[d]-1)))
df=pd.DataFrame(rows,columns=['date','s','sig','h','fut']); df.to_csv('scripts/miner_2_20330624_asymmetry_signal.csv',index=False)
print('dates',len(dates),'used',df.date.nunique(),'avg_n',df.groupby('date').size().mean(),'coverage',len(df)/(len(dates)*15*4))
for h,g in df.groupby('h'):
 ic=g.groupby('date').apply(lambda x:x.sig.corr(x.fut),include_groups=False).dropna()
 print('H',h,'IC',round(g.sig.corr(g.fut),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'n',len(ic))
 for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
  z=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]; print(a,b,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),5) if len(z)>1 else None)
