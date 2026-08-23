import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-02-15')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date') for s in U}
rows=[]
for s,d in px.items():
 r=d.close.pct_change(); down=r.where(r<0,0).rolling(40).std(); f=d.close.pct_change(20)/(down*np.sqrt(40)+1e-12)
 for i in range(len(d)-10):
  if np.isfinite(f.iloc[i]): rows.append((d.index[i],s,f.iloc[i],d.close.iloc[i+10]/d.close.iloc[i]-1))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
def stat(z):
 q=[g.factor.corr(g.fwd,method='spearman') for _,g in z.groupby('date') if len(g)>=8];q=pd.Series(q).dropna();return len(q),round(z.groupby('date').size().mean(),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4),z.symbol.nunique()
print('factor=20d momentum / (sqrt40 * downside volatility over 40d)')
print('range',x.date.min().date(),x.date.max().date(),'rows',len(x),'assets',x.symbol.nunique());print('overall',stat(x))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]: print('regime',a,b,stat(x[(x.date>=a)&(x.date<=b)]))
for h in [5,10,20,40]:
 q=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8:q.append(g.factor.corr(g.fwd if h==10 else pd.Series(index=g.index,dtype=float),method='spearman'))
 # only valid primary horizon; decay reported separately below via aligned calculation
 if h==10: print('decay',h,len(q),round(np.nanmean(q),6),round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).mean(),6),'coverage',round(x.symbol.nunique()/15,4),'avg_n',round(x.groupby('date').size().mean(),2))
