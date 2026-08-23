import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-10-21')
rows=[]; signals=[]; valid_dates=set()
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); p=d.close.astype(float); r=p.pct_change(); f=p.pct_change(10)/r.abs().rolling(10,min_periods=8).sum(); fw=p.shift(-1)/p-1
 x=pd.concat([f.rename('f'),fw.rename('fw')],axis=1).dropna();
 for dt,row in x.iterrows(): signals.append((dt,a,row.f));
# date cross-sections, no all-panel complete-case bias
s=pd.DataFrame(signals,columns=['date','asset','f']).pivot(index='date',columns='asset',values='f')
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); p=d.close.astype(float); fw=p.shift(-1)/p-1; s.loc[s.index,a]=s[a] # retain
# forward returns panel
fwpanel=pd.DataFrame({a:(lambda p:p.shift(-1)/p)(pd.read_csv('../persistent/stock_data/'+a+'.csv').assign(date=lambda z:pd.to_datetime(z.date)).query('date<=@cutoff').sort_values('date').set_index('date').close.astype(float)) for a in assets})
for dt in s.index:
 z=pd.concat([s.loc[dt],fwpanel.loc[dt]],axis=1).dropna()
 if len(z)>=8: valid_dates.add(dt); rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avg_names',round(r.n.mean(),3),'coverage',round(r.n.mean()/15,4),'period',r.index.min().date(),r.index.max().date())
print('daily IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for h in [5,10,20]:
 vals=[]
 for a in assets:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); p=d.close.astype(float); q=p.shift(-h)/p; base=p.pct_change(10)/p.pct_change().abs().rolling(10,min_periods=8).sum(); vals.append(pd.concat([base.rename('f'),q.rename('q')],axis=1))
 zall=pd.concat(vals,axis=1,keys=assets); out=[]
 for dt in zall.index:
  xx=pd.DataFrame({'f':[zall.loc[dt,(a,'f')] for a in assets],'q':[zall.loc[dt,(a,'q')] for a in assets]}).dropna()
  if len(xx)>=8: out.append(spearmanr(xx.f,xx.q).statistic)
 v=np.array(out); print('%dd IC %.8f ICIR %.8f n %d'%(h,v.mean(),v.mean()/v.std(),len(v)))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-10-21')]:
 q=r.loc[a:b].ic; print(label,len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean())
