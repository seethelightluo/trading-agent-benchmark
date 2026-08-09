import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# volume-confirmed short reversal: negative 5d return, scaled by abnormal log-volume
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: continue
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.sort_values('date')
 d['r1']=d.close.pct_change(); d['r5']=d.close.pct_change(5)
 lv=np.log(d.volume.replace(0,np.nan))
 d['vz']=(lv-lv.rolling(60,min_periods=30).mean())/lv.rolling(60,min_periods=30).std()
 # volume shock tends to mark exhaustion: reversal magnitude strengthened by abnormal volume
 d['f']= -d.r5*(1+d.vz.clip(-1,3).fillna(0).abs())
 d['f2']= -d.r5*d.vz.clip(-2,2).fillna(0)
 d['fr']= -d.r5
 d['next']=d.close.shift(-1)/d.close-1
 rows.append(d[['date','f','f2','fr','next']].assign(symbol=s))
x=pd.concat(rows).dropna(subset=['next'])
def eval(col,h=1):
 z=x.dropna(subset=[col]).copy(); vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8: vals.append(g[col].corr(g['next']))
 a=pd.Series(vals).dropna(); return len(a),a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()
print('symbols',x.symbol.nunique(),'rows',len(x),'dates',x.date.nunique())
for c in ['f','f2','fr']:
 print(c,'daily n mean IC ICIR hit',eval(c))
# 5d decay using forward close 5d, aligned per asset
for h in [5,10]:
 xx=[]
 for s in U:
  d=get_stock_daily_data(s,days=3000)
  if d is None: continue
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date'); d['r5']=d.close.pct_change(5)
  lv=np.log(d.volume.replace(0,np.nan)); z=(lv-lv.rolling(60,min_periods=30).mean())/lv.rolling(60,min_periods=30).std()
  d['f']=-d.r5*(1+z.clip(-1,3).fillna(0).abs()); d['fw']=d.close.shift(-h)/d.close-1
  xx.append(d[['date','f','fw']])
 q=pd.concat(xx).dropna(); aa=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: aa.append(g.f.corr(g.fw))
 a=pd.Series(aa).dropna(); print('h',h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
