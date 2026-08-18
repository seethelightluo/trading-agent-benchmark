import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
syms=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index();r=P.pct_change();dd=r.clip(upper=0).rolling(40).std()*np.sqrt(40);f=-(r.rolling(60).sum()/dd)
ics=[];ns=[];tos=[]
for i in range(150,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8:continue
 a=f.iloc[i][n];y=P.iloc[i+10][n]/P.iloc[i][n]-1;ics.append((P.index[i],a.corr(y,method='spearman')));ns.append(len(n))
 if i>150:tos.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][n].rank(pct=True))))
ser=pd.Series(dict(ics)).dropna()
for l,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]:print(l,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
print('avg_valid',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(syms),4),'turnover',round(np.mean(tos),4),'period',P.index[0],P.index[-1])
for j,z in enumerate(np.array_split(ser,4),1):print('block',j,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}).to_csv('scripts/miner_3_20350216_downside_reversal_signal.csv',index=False)
