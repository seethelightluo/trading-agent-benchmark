import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000);d.date=pd.to_datetime(d.date);C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index();R=np.log(P).diff();res=R.sub(R.median(axis=1),axis=0)
def f(i):
 tr=res.iloc[i-39:i+1].sum();v=res.iloc[i-59:i+1].std()*np.sqrt(40);bread=(tr>0).mean()
 return -tr/(v+1e-12)*(0.25+0.75*bread)
for h in [5,10,20]:
 out=[]
 for i in range(70,len(P)-h):
  x=pd.DataFrame({'f':f(i),'y':P.iloc[i+h]/P.iloc[i]-1}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(x)>=8 and x.f.nunique()>1:out.append(x.f.corr(x.y,method='spearman'))
 a=np.array(out);print('horizon',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),5))
# signal artifact at primary 10d
rows=[]
for i in range(70,len(P)-10):
 for s,v in f(i).items():rows.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
S=pd.DataFrame(rows).pivot(index='date',columns='symbol',values='signal');print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6));pd.DataFrame(rows).to_csv('scripts/miner_3_20310109_inverse_40d_trend_signal.csv',index=False)
