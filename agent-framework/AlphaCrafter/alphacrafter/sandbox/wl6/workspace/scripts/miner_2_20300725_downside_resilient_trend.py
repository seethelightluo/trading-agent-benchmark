import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,3000) for s in U}
def signal(df):
 r=df.close.pct_change(); down=r.where(r<0,0.0); dd=down.pow(2).rolling(30).mean().pow(.5)*np.sqrt(252)
 return df.close.pct_change(30)/(dd+1e-8)*(r.rolling(30).mean().gt(0).astype(float)*.5+.5)
rows=[]
for s,df in D.items():
 if df is None or len(df)<100: continue
 df=df.copy(); df['date']=pd.to_datetime(df.date); sig=signal(df)
 for i in range(60,len(df)-10):
  if pd.notna(sig.iloc[i]): rows.append((df.date.iloc[i],s,float(sig.iloc[i]),float(df.close.iloc[i+1+9]/df.close.iloc[i+1]-1)))
x=pd.DataFrame(rows,columns=['date','symbol','f','fr']); out=[]
for d,g in x.groupby('date'):
 if len(g)>=8: out.append([d,g.f.corr(g.fr,method='spearman'),len(g)])
o=pd.DataFrame(out,columns=['date','ic','n']).dropna(); print('dates',len(o),'avg_n',o.n.mean(),'coverage',x.groupby('date').size().mean()/15); print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1),'hit',(o.ic>0).mean())
for y,g in o.groupby(o.date.dt.year): print(y,round(g.ic.mean(),5),len(g))
ranks=x.assign(rank=x.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').sort_index(); print('turnover',ranks.diff().abs().mean().mean())
for h in [5,20]:
 z=[]
 for s,df in D.items():
  if df is None: continue
  ss=signal(df)
  for i in range(60,len(df)-h):
   if pd.notna(ss.iloc[i]): z.append((df.date.iloc[i],float(ss.iloc[i]),float(df.close.iloc[i+1+h-1]/df.close.iloc[i+1]-1)))
 q=pd.DataFrame(z,columns=['d','f','r']); vals=[g.f.corr(g.r,method='spearman') for _,g in q.groupby('d') if len(g)>=8]; vals=pd.Series(vals).dropna(); print('h',h,'ic',vals.mean(),'icir',vals.mean()/vals.std(ddof=1))
