import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 df=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: df=fn(s,days=3000)
  except Exception: df=None
  if df is not None and len(df)>=120: break
 if df is None or len(df)<120: continue
 d=df[['date','close']].copy().sort_values('date'); d['r']=d.close.pct_change()
 r20=d.close/d.close.shift(20)-1; r60=d.close/d.close.shift(60)-1
 down=d.r.where(d.r<0).rolling(30).std()*np.sqrt(20); vol60=d.r.rolling(60).std()*np.sqrt(60)
 conf=.5+.5*np.tanh(r60/(vol60+1e-12)); fac=(r20/(down*np.sqrt(20)+1e-8)*conf).shift(1)
 for date,f,y1,y5 in zip(d.date,fac,d.close.shift(-1)/d.close-1,d.close.shift(-5)/d.close-1): rows.append((date,s,f,y1,y5))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd1','fwd5']); obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','fwd1','fwd5'])
 if len(g)>=8: obs.append([dt,len(g),g.factor.corr(g.fwd1,method='spearman'),g.factor.corr(g.fwd5,method='spearman')])
o=pd.DataFrame(obs,columns=['date','n','ic1','ic5']).sort_values('date'); o.date=pd.to_datetime(o.date)
for c in ['ic1','ic5']:
 q=o.dropna(subset=[c]); mu=q[c].mean(); sd=q[c].std(ddof=1); print(c,'mean',mu,'ir',mu/sd*np.sqrt(len(q)),'hit',(q[c]>0).mean())
print('dates',len(o),'avg_n',o.n.mean(),'range',o.date.min(),o.date.max())
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=o[(o.date.dt.year>=a)&(o.date.dt.year<=b)].dropna(subset=['ic1']); print('regime',a,b,'n',len(q),'ic',q.ic1.mean(),'ir',q.ic1.mean()/q.ic1.std(ddof=1)*np.sqrt(len(q)))
p=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor'); turn=[]
for i in range(1,len(p)):
 a=p.iloc[i-1].rank(pct=True); b=p.iloc[i].rank(pct=True); z=a.index.intersection(b.index)
 if len(z)>=8: turn.append((a[z]-b[z]).abs().mean())
print('rank_turnover',np.mean(turn),'coverage',x.factor.notna().mean(),'assets',x.symbol.nunique())
x.to_csv('scripts/miner_3_20270402_continuous_trend_confirmation_signal.csv',index=False)
