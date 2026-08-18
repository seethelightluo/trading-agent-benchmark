import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
p=pd.DataFrame(P).sort_index(); r=p.pct_change();
# peer breadth: leave-one-out fraction of other assets with positive 5d return, centered
r5=p.pct_change(5); rows=[]
for i,d in enumerate(p.index):
 if i<10 or i+1>=len(p.index): continue
 vals=r5.iloc[i]
 if vals.notna().sum()<8: continue
 for j,s in enumerate(U):
  peers=vals.drop(s).dropna()
  if len(peers)<7: continue
  f=(peers>0).mean()-0.5
  fw=p[s].iloc[i+1]/p[s].iloc[i]-1
  if np.isfinite(fw): rows.append((d,s,f,fw))
a=pd.DataFrame(rows,columns=['date','s','f','fw'])
ics=[]
for d,g in a.groupby('date'):
 if len(g)>=8: ics.append(spearmanr(g.f,g.fw).statistic)
ic=np.array(ics)
print('peer_breadth dates',len(ic),'avgN',a.groupby('date').size().mean(),'coverage',len(a)/(len(p.index)*15),'IC',np.nanmean(ic),'ICIR',np.nanmean(ic)/np.nanstd(ic,ddof=1),'hit',np.mean(ic>0))
for h in [5,10]:
 z=[]
 for d,g in a.groupby('date'):
  # recreate fw from each symbol at date
  fs=[]; ys=[]
  for _,q in g.iterrows():
   i=p.index.get_loc(d); y=p[q.s].iloc[i+h]/p[q.s].iloc[i]-1 if i+h<len(p) else np.nan
   if np.isfinite(y):fs.append(q.f);ys.append(y)
  if len(fs)>=8:z.append(spearmanr(fs,ys).statistic)
 z=np.array(z);print(h,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
print('turnover',a.sort_values(['s','date']).groupby('s').f.apply(lambda x:x.diff().abs().mean()).mean())
for y,g in a.groupby(a.date.dt.year):
 z=[spearmanr(x.f,x.fw).statistic for _,x in g.groupby('date') if len(x)>=8];print(y,len(z),np.nanmean(z))
