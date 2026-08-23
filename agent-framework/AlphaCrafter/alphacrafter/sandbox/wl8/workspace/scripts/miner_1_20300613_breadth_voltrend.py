import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-13')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
r20=p.pct_change(20); r60=p.pct_change(60)
breadth=(r20>0).mean(axis=1); reg=pd.Series(np.where(breadth>=0.5,1.0,-1.0),index=p.index)
f=(r20/vol).mul(reg,axis=0) - 0.20*(r60/vol)
f=f.sub(f.median(axis=1),axis=0).clip(-8,8)
def run(h):
 out=[]; ns=[]; ds=[]
 for i in range(len(p)-h):
  if p.index[i] < pd.Timestamp('2026-01-01') or p.index[i+h]>cut: continue
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   out.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(p.index[i])
 return np.asarray(out),np.asarray(ds),ns
ics,dates,ns=run(10); ic=ics.mean(); icir=ic/ics.std(ddof=1)
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print({'factor':'breadth_conditioned_volatility_normalized_trend','dates':len(ics),'start':str(dates[0].date()),'end':str(dates[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'ic':float(ic),'icir':float(icir),'hit':float((ics>0).mean()),'turnover':float(turn)})
for h in [5,20]:
 a,_,_=run(h); print('decay',h,float(a.mean()),float(a.mean()/a.std(ddof=1)),len(a))
for n in [180,360]:
 a=ics[-n:]; print('recent',n,float(a.mean()),float(a.mean()/a.std(ddof=1)),len(a))
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_1_20300613_breadth_voltrend_ic.csv',index=False)
f.loc[dates].stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300613_breadth_voltrend_signal.csv',index=False)
