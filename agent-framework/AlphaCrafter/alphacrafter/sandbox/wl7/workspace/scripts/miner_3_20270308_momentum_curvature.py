import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# medium-vs-long momentum curvature, lagged by one completed session
rets={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); d=d.loc[:'2027-03-08']
 r=d.close.pct_change()
 rets[s]=pd.DataFrame({'r':r,'f':(d.close.pct_change(20)-d.close.pct_change(60))/(r.rolling(20).std()*np.sqrt(20)),'y':r.shift(-1)})
all_dates=sorted(set().union(*[set(x.index) for x in rets.values()]))
ics=[]; rows=[]; sig=[]
for dt in all_dates:
 vals=[]; ys=[]
 for s,x in rets.items():
  if dt in x.index:
   a=x.loc[dt]
   if pd.notna(a.f) and pd.notna(a.y): vals.append(a.f); ys.append(a.y)
 if len(vals)>=8:
  ic=spearmanr(vals,ys).statistic
  ics.append(ic); rows.append((dt,ic,len(vals)))
  sig.append((dt, np.nanstd(vals)))
a=np.array(ics)
print('dates',len(a),'avg_n',np.mean([x[2] for x in rows]),'ic',np.nanmean(a),'icir',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'coverage',np.mean([x[2] for x in rows])/15)
for h in [5,10,20]:
 z=[]
 for dt in all_dates:
  vs=[]; yy=[]
  for s,x in rets.items():
   if dt in x.index:
    q=x.loc[dt]; y= x.r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt] if False else None
    # calculate forward h return from close t to close t+h
    try: y=x.r.shift(-1).rolling(h).sum().loc[dt]
    except: y=np.nan
    if pd.notna(q.f) and pd.notna(y): vs.append(q.f); yy.append(y)
  if len(vs)>=8:z.append(spearmanr(vs,yy).statistic)
 print('h',h,'n',len(z),'ic',np.nanmean(z),'icir',np.nanmean(z)/np.nanstd(z,ddof=1))
# rank turnover using dates with signals
prev=None; turns=[]
for dt in all_dates:
 vals={s:x.loc[dt,'f'] for s,x in rets.items() if dt in x.index and pd.notna(x.loc[dt,'f'])}
 if len(vals)>=8:
  rank=pd.Series(vals).rank(pct=True)
  if prev is not None:
   common=rank.index.intersection(prev.index); turns.append(np.mean(np.abs(rank[common]-prev[common])))
  prev=rank
print('rank_turnover',np.mean(turns),'first',rows[0][0],'last',rows[-1][0])
# artifact
out=[]
for dt in all_dates:
 for s,x in rets.items():
  if dt in x.index and pd.notna(x.loc[dt,'f']):out.append({'date':dt,'symbol':s,'signal':x.loc[dt,'f']})
pd.DataFrame(out).to_csv('scripts/miner_3_20270308_momentum_curvature_signal.csv',index=False)
