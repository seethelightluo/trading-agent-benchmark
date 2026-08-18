import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=[s for s in get_account_dict().get('watch_list',[]) if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<200:d=get_index_daily_data(s,5000)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); lr=np.log(p).diff(); down=lr.clip(upper=0).rolling(40).std();
# downside-adjusted medium reversal, lagged
fac=(-lr.rolling(20).sum()/down).shift(1)
for h in [1,5,10,20,40]:
 fw=np.log(p).shift(-h)-np.log(p); x=[]; ns=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(x).dropna(); print(h,len(x),round(np.mean(ns),2),f'{x.mean():.6f}',f'{x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}',f'{(x>0).mean():.4f}')
fac.to_csv('scripts/miner_2_20340721_downside_reversal_signal.csv')
