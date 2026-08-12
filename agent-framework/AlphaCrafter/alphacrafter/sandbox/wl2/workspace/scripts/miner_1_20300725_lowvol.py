import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_account_dict
syms=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>80:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); rows=[]
for i,dt in enumerate(p.index):
 if i<31 or i+10>=len(p): continue
 # inverse volatility, mildly rewarded for positive recent return, all lagged
 v=r.iloc[i-20:i].std(); mom=r.iloc[i-20:i].sum()
 sig=-(v.rank(pct=True)) + 0.25*mom.rank(pct=True)
 for s in syms:
  if s in sig.index and pd.notna(sig[s]) and pd.notna(p.iloc[i+1][s]): rows.append((dt,s,float(sig[s]),float(p.iloc[i+1][s]/p.iloc[i][s]-1)))
d=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).set_index('date')
ics=d.groupby(level=0).apply(lambda z:z.factor.corr(z.fwd) if len(z)>=8 else np.nan).dropna()
print('dates',len(ics),'avg_n',d.groupby(level=0).size().mean(),'coverage',len(d)/(len(ics)*len(syms)))
print('h1 IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),(ics>0).mean()))
for h in [5,10]:
 a=[]
 for dt,g in d.groupby(level=0):
  i=p.index.get_loc(dt)
  if i+h>=len(p):continue
  for _,z in g.iterrows():
   s=z.symbol
   if pd.notna(p.iloc[i+h][s]):a.append((dt,z.factor,p.iloc[i+h][s]/p.iloc[i][s]-1))
 q=pd.DataFrame(a,columns=['date','factor','fwd']).groupby('date').apply(lambda z:z.factor.corr(z.fwd) if len(z)>=8 else np.nan).dropna()
 print('h%d IC %.6f ICIR %.6f hit %.4f dates %d'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
d.reset_index()[['date','symbol','factor']].to_csv('scripts/miner_1_20300725_lowvol_signal.csv',index=False)
