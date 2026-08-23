import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=pd.Timestamp('2027-03-02')].copy(); d['r']=d.close.pct_change()
 r20=d.close/d.close.shift(20)-1; r60=d.close/d.close.shift(60)-1; down=d.r.where(d.r<0).rolling(30).std()*np.sqrt(20); vol60=d.r.rolling(60).std()*np.sqrt(60)
 fac=(r20/(down*np.sqrt(20)+1e-8)*(.5+.5*np.tanh(r60/(vol60+1e-12)))).shift(1)
 for date,x,y,z in zip(d.date,fac,d.close.shift(-1)/d.close-1,d.close.shift(-10)/d.close-1): rows.append((date,s,x,y,z))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd1','fwd10']); obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','fwd1','fwd10'])
 if len(g)>=8: obs.append([dt,len(g),g.factor.corr(g.fwd1,method='spearman'),g.factor.corr(g.fwd10,method='spearman')])
o=pd.DataFrame(obs,columns=['date','n','ic1','ic10']); o['date']=pd.to_datetime(o['date']); o=o.dropna().sort_values('date'); print('dates',len(o),'avg_n',o.n.mean(),'range',o.date.min(),o.date.max())
for c in ['ic1','ic10']:
 mu=o[c].mean(); sd=o[c].std(ddof=1); print(c,'mean',mu,'ir',mu/sd*np.sqrt(len(o)),'hit',(o[c]>0).mean())
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=o[(o.date.dt.year>=a)&(o.date.dt.year<=b)]; print('regime',a,b,'n',len(q),'ic1',q.ic1.mean())
p=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor'); turn=[]
for i in range(1,len(p)):
 a=p.iloc[i-1].rank(pct=True); b=p.iloc[i].rank(pct=True); common=a.index.intersection(b.index)
 if len(common)>=8: turn.append((a[common]-b[common]).abs().mean())
print('rank_turnover',np.mean(turn),'coverage',x.factor.notna().mean()); x.to_csv('scripts/miner_3_20270302_continuous_trend_confirmation_signal.csv',index=False)
