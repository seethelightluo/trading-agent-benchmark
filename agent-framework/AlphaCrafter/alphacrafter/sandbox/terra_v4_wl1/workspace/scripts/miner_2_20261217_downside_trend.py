import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float); P[s]=d[d.index<=cut]
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); mom=R.rolling(40,min_periods=25).sum(); down=R.where(R<0).rolling(40,min_periods=25).std()*np.sqrt(40); F=mom.div(down.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); rows=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   c=z.f.corr(z.y,method='spearman')
   if pd.notna(c): rows.append((dt,c,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); a['date']=pd.to_datetime(a['date']); q=a.ic
 print('H',h,'dates',len(q),'avgN',round(a.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1: print('years',[(int(y),len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),4)) for y,g in q.groupby(a.date.dt.year)])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5)); F.to_csv('scripts/miner_2_20261217_downside_trend_signal.csv')
