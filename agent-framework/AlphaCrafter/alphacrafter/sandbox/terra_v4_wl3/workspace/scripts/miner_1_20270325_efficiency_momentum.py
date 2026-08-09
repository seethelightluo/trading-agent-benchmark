import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-03-24')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 d=d.loc[:end]; r=d.close.pct_change()
 # Efficiency of medium-term trend: net 20-day return divided by path length
 net=d.close.pct_change(20)
 path=r.abs().rolling(20).sum()
 xs[s]=pd.DataFrame({'f':net/path,'ret':r})
all_dates=sorted(set().union(*[x.index for x in xs.values()]))
rows=[]; sig=[]
for dt in all_dates:
 vals={s:xs[s].loc[dt,'f'] for s in U if dt in xs[s].index and pd.notna(xs[s].loc[dt,'f'])}
 if len(vals)<8: continue
 # factor known at close dt, forward return next trading observation per asset
 for s,v in vals.items():
  ix=xs[s].index.get_loc(dt)
  if ix+1<len(xs[s]):
   fr=xs[s].iloc[ix+1].ret
   if pd.notna(fr): rows.append((dt,s,float(v),float(fr)))
  sig.append((dt,s,float(v)))
df=pd.DataFrame(rows,columns=['date','symbol','f','fr'])
ics=df.groupby('date').apply(lambda z: spearmanr(z.f,z.fr).statistic if len(z)>=8 else np.nan).dropna()
# turnover based rank ordering / normalized signal changes
sd=pd.DataFrame(sig,columns=['date','symbol','f']); sd['rank']=sd.groupby('date').f.rank(pct=True)
wide=sd.pivot(index='date',columns='symbol',values='rank').sort_index()
to=wide.diff().abs().mean(axis=1).mean()
print('dates',len(ics),'rows',len(df),'avgN',df.groupby('date').size().mean(),'coverage',len(df)/(len(ics)*15),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'turnover',to,'first',ics.index.min(),'last',ics.index.max())
# decay 5/10 using forward compounded aligned per symbol
for h in [5,10]:
 rr=[]
 for dt in all_dates:
  vals={s:xs[s].loc[dt,'f'] for s in U if dt in xs[s].index and pd.notna(xs[s].loc[dt,'f'])}
  a=[];b=[]
  for s,v in vals.items():
   ix=xs[s].index.get_loc(dt)
   if ix+h<len(xs[s]):
    fwd=xs[s].iloc[ix+1:ix+h+1].ret.add(1).prod()-1
    if pd.notna(fwd):a.append(v);b.append(fwd)
  if len(a)>=8: rr.append(spearmanr(a,b).statistic)
 rr=pd.Series(rr).dropna();print('h',h,'n',len(rr),'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1))
pd.DataFrame(sig,columns=['date','symbol','f']).to_csv('scripts/miner_1_20270325_efficiency_momentum_signal.csv',index=False)
