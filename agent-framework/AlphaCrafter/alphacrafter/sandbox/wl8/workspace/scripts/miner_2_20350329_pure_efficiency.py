import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f):continue
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();c=d.close.astype(float);r=c.pct_change(); path=r.abs().rolling(30,min_periods=22).sum(); sig=(c.pct_change(30)/(path+1e-12)).shift(1);P[s]=pd.DataFrame({'f':sig,'c':c})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10);z=pd.concat([x.f,y.rename('y')],axis=1).dropna();z['s']=s;rows.append(z.reset_index())
a=pd.concat(rows,ignore_index=True);out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q):out.append((dt,q,len(g)))
i=pd.DataFrame(out,columns=['date','ic','n']);q=i.ic; rank=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('assets',len(P),'dates',len(i),'avgN',i.n.mean(),'coverage',len(a)/(len(set(a.date))*15));print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean()/2);a.to_csv('scripts/miner_2_20350329_pure_efficiency_panel.csv',index=False);i.to_csv('scripts/miner_2_20350329_pure_efficiency_ic.csv',index=False)
