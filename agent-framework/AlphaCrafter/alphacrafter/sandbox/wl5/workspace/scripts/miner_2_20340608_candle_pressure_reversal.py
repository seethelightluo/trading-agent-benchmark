import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2034-06-07')
frames={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=end].set_index('date')
 # Candle pressure: close location in daily range; aggregate recent pressure, penalizing persistent one-way moves via inverse vol
 rng=(d.high-d.low).replace(0,np.nan)
 loc=((2*d.close-d.high-d.low)/rng).clip(-1,1)
 ret=d.close.pct_change()
 pressure=loc.rolling(10,min_periods=8).mean()
 vol=ret.rolling(20,min_periods=15).std()
 # reversal of recent directional candle pressure, volatility scaled
 frames[s]=pd.DataFrame({'f':-pressure/(vol*np.sqrt(10)), 'close':d.close})
all_dates=sorted(set.intersection(*[set(x.index) for x in frames.values()]))
ics=[]; rows=[]; sigrows=[]
for dt in all_dates:
 vals=[]; fw=[]; sy=[]
 for s in symbols:
  x=frames[s]
  if dt not in x.index: continue
  # strictly future 10 rows after dt
  pos=x.index.get_loc(dt)
  if pos+10>=len(x): continue
  a=x.iloc[pos].f; b=x.iloc[pos+10].close/x.iloc[pos].close-1
  if np.isfinite(a) and np.isfinite(b): vals.append(a);fw.append(b);sy.append((s,a))
 if len(vals)>=8:
  ic=spearmanr(vals,fw).statistic
  if np.isfinite(ic):
   ics.append(ic); rows.append((dt,ic,len(vals)))
   for s,a in sy: sigrows.append((dt,s,a))
ser=pd.Series([r[1] for r in rows],index=[r[0] for r in rows])
print('dates',len(ser),'universe',15,'meanN',np.mean([r[2] for r in rows]),'coverage',sum(r[2] for r in rows)/(len(rows)*15))
print('IC',ser.mean(),'ICIR_ann',ser.mean()/ser.std(ddof=1)*np.sqrt(252),'hit',(ser>0).mean(),'std',ser.std())
print('regimes')
for a,b in [('2026','2027'),('2028','2029'),('2030','2032'),('2033','2034')]:
 z=ser[(ser.index.year>=int(a))&(ser.index.year<=int(b))];print(a,b,len(z),z.mean())
# decay proxy evaluate alternative horizons from same factor
for h in [5,10,20]:
 q=[]
 for dt in all_dates:
  va=[];fb=[]
  for s in symbols:
   x=frames[s]
   if dt not in x.index:continue
   p=x.index.get_loc(dt)
   if p+h>=len(x):continue
   aa=x.iloc[p].f; bb=x.iloc[p+h].close/x.iloc[p].close-1
   if np.isfinite(aa) and np.isfinite(bb):va.append(aa);fb.append(bb)
  if len(va)>=8:q.append(spearmanr(va,fb).statistic)
 print('h',h,'IC',np.nanmean(q),'dailyICIR',np.nanmean(q)/np.nanstd(q,ddof=1),'annICIR',np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(252),'n',len(q))
# turnover rank changes
wide=pd.DataFrame(index=all_dates,columns=symbols,dtype=float)
for dt,s,a in sigrows:wide.loc[dt,s]=a
ranks=wide.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print('rank_turnover',turnover)
out='scripts/miner_2_20340608_candle_pressure_reversal_signal.csv';pd.DataFrame(sigrows,columns=['date','symbol','signal']).to_csv(out,index=False);print('artifact',out)
