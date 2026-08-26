import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# curvature: recent 20d return minus prior 40d return, scaled by 60d vol; lag one day
frames={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d)>200:
        d=d[['date','close']].copy().drop_duplicates('date').set_index('date').sort_index()
        r=d.close.pct_change()
        d['f']=((d.close.shift(1)/d.close.shift(21)-1) - (d.close.shift(21)/d.close.shift(61)-1)) / (r.rolling(60).std().shift(1)*np.sqrt(20))
        d['f']=d.f.replace([np.inf,-np.inf],np.nan)
        frames[s]=d
all_dates=sorted(set().union(*[set(x.index) for x in frames.values()]))
rows=[]
for dt in all_dates:
    vals=[]; fw=[]
    for s,d in frames.items():
        if dt not in d.index: continue
        i=d.index.get_loc(dt)
        if i+10>=len(d): continue
        v=d.iloc[i].f; y=d.close.iloc[i+10]/d.close.iloc[i]-1
        if np.isfinite(v) and np.isfinite(y): vals.append(v); fw.append(y)
    if len(vals)>=8:
        rows.append((dt,len(vals),pd.Series(vals).rank().corr(pd.Series(fw).rank())))
res=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('candidate=curvature20v40_scaled60; dates',len(res),'avgN',res.n.mean(),'start',res.index.min(),'end',res.index.max())
print('IC %.6f ICIR %.6f hit %.4f' % (res.ic.mean(),res.ic.mean()/res.ic.std(),(res.ic>0).mean()))
for h in [1,5,10,20]:
 rr=[]
 for dt in all_dates:
  vals=[]; fw=[]
  for s,d in frames.items():
   if dt not in d.index: continue
   i=d.index.get_loc(dt)
   if i+h>=len(d): continue
   v=d.iloc[i].f; y=d.close.iloc[i+h]/d.close.iloc[i]-1
   if np.isfinite(v) and np.isfinite(y): vals.append(v);fw.append(y)
  if len(vals)>=8: rr.append(pd.Series(vals).rank().corr(pd.Series(fw).rank()))
 rr=pd.Series(rr)
 print('h',h,'IC %.6f ICIR %.6f n %d'%(rr.mean(),rr.mean()/rr.std(),len(rr)))
# coverage and rank turnover on common dates
valid=[]
for dt in all_dates:
 a=[]
 for s,d in frames.items():
  if dt in d.index and np.isfinite(d.loc[dt,'f']): a.append((s,d.loc[dt,'f']))
 if len(a)>=8: valid.append((dt,dict(a)))
turn=[]
for (_,a),(_,b) in zip(valid[:-1],valid[1:]):
 common=set(a)&set(b)
 if common:
  ra=pd.Series(a).rank(pct=True); rb=pd.Series(b).rank(pct=True); turn.append(np.mean([abs(ra[x]-rb[x]) for x in common]))
print('coverage_calendar %.4f avg_valid %.2f rank_turnover_proxy %.4f'%(len(valid)/len(all_dates),np.mean([len(x[1]) for x in valid]),np.mean(turn) if turn else np.nan))
# recent regime slices
for n in [250,500,750]:
 x=res.tail(n);print('recent',n,'IC %.6f ICIR %.6f dates %d'%(x.ic.mean(),x.ic.mean()/x.ic.std(),len(x)))
# artifact
out=[]
for dt,a in valid:
 for s,v in a.items(): out.append({'date':dt,'symbol':s,'signal':v})
pd.DataFrame(out).to_csv('scripts/miner_2_20340724_curvature20v40_signal.csv',index=False)
