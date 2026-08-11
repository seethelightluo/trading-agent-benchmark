import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
end=pd.Timestamp('2026-07-15'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p):p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index().loc[:end]
cs=pd.concat([L(s).close.pct_change(1).rename(s) for s in A],axis=1); breadth=cs.gt(0).sum(axis=1)/cs.notna().sum(axis=1)
# momentum is trusted only when breadth confirms its direction; demeaned breadth avoids a constant shift
confirm=(breadth.rolling(5,min_periods=3).mean()-0.5).shift(1)
for h in [1,5,10]:
 rows=[]
 for s in A:
  d=L(s); f=(d.close.pct_change(20)*confirm).rename('f'); y=(d.close.shift(-h)/d.close-1).rename('y');q=pd.concat([f,y],axis=1).dropna();rows += [(t,float(a),float(b)) for t,a,b in zip(q.index,q.f,q.y)]
 x=pd.DataFrame(rows,columns=['date','f','y']);v=[];nn=[]
 for t,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic);nn.append(len(g))
 v=np.array(v);print('h',h,'dates',len(v),'avg_n',np.mean(nn),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
if True:
 h=10;rows=[]
 for s in A:
  d=L(s);q=pd.concat([(d.close.pct_change(20)*confirm).rename('f'),(d.close.shift(-h)/d.close-1).rename('y')],axis=1).dropna();rows += [(t,a,b) for t,a,b in zip(q.index,q.f,q.y)]
 x=pd.DataFrame(rows,columns=['date','f','y']);x['yr']=x.date.dt.year
 for yr,g in x.groupby('yr'):
  v=[]
  for t,gg in g.groupby('date'):
   if len(gg)>=8 and gg.f.nunique()>1 and gg.y.nunique()>1:v.append(spearmanr(gg.f,gg.y).statistic)
  print('year',yr,'IC',np.mean(v),'n',len(v))
