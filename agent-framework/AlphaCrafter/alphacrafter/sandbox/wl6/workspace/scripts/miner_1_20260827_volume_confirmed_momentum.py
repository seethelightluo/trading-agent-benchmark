import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
# volume-confirmed momentum: cumulative return weighted by relative log-volume trend
C={}; V={}
for s in U:
 d=pd.read_csv(f'{b}/{s}.csv'); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); C[s]=d.close; V[s]=d.volume.replace(0,np.nan)
P=pd.DataFrame(C).sort_index(); Vol=pd.DataFrame(V).reindex(P.index); R=P.pct_change(); lv=np.log(Vol)
# positive volume trend confirms direction; bounded multiplier prevents extreme volume outliers
vt=(lv.rolling(20,min_periods=15).mean()-lv.rolling(60,min_periods=40).mean())/lv.rolling(60,min_periods=40).std()
vt=vt.clip(-3,3)
F=R.rolling(20,min_periods=15).sum()*(1+0.25*vt)
rows=[]
for s in U:
 for dt in P.index:
  rows.append((dt,s,F.at[dt,s],R[s].shift(-1).at[dt]))
A=pd.DataFrame(rows,columns=['date','asset','f','y']).dropna(); out=[]
for dt,g in A.groupby('date'):
 if len(g)>=8: out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
i=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').ic
print('dates',len(i),'avg_names',round(A.groupby('date').size().loc[i.index].mean(),2),'IC',round(i.mean(),5),'ICIR',round(i.mean()/i.std(ddof=1),5),'hit',round((i>0).mean(),4),'coverage',round(len(A)/(len(P)*15),4))
for h in [5,10]:
 y=P.pct_change(h).shift(-h) # return t to t+h, aligned at t
 B=pd.DataFrame([(dt,s,F.at[dt,s],y[s].at[dt]) for s in U for dt in P.index],columns=['date','asset','f','y']).dropna(); q=[]
 for dt,g in B.groupby('date'):
  if len(g)>=8:q.append(spearmanr(g.f,g.y).statistic)
 q=pd.Series(q); print('h',h,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
for yr in range(2020,2027):
 q=i[i.index.year==yr]
 if len(q):print('regime',yr,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5))
# rank turnover
r=F.rank(axis=1,pct=True); print('turnover',round((r.diff().abs().mean(axis=1).dropna()).mean(),4))
