import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2027-04-07');rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);x=x[x.date<=END].sort_values('date');x['sig']=-(x.close/x.open-1);x['fwd']=x.close.shift(-1)/x.close-1;x=x.dropna(subset=['sig','fwd']);x['symbol']=s;rows.append(x[['date','symbol','sig','fwd']])
z=pd.concat(rows);a=[];n=[]
for d,g in z.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:a.append(spearmanr(g.sig,g.fwd).statistic);n.append(len(g))
a=np.array(a);print('dates',len(a),'rows',len(z),'avg_names',np.mean(n),'coverage',len(z)/(15*z.date.nunique()),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for label,cut in [('recent',z.date>=pd.Timestamp('2026-04-01')),('2027',z.date.dt.year==2027)]:
 q=[spearmanr(h.sig,h.fwd).statistic for _,h in z[cut].groupby('date') if len(h)>=8 and h.sig.nunique()>1 and h.fwd.nunique()>1];print(label,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1) if len(q)>1 else None)
z[['date','symbol','sig']].to_csv('scripts/miner_3_20270408_intraday_reversal_signal.csv',index=False)
