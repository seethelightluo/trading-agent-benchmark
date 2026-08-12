import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list');P={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100:d=get_index_daily_data(s,1800)
 P[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index();R=P.pct_change();m=R.iloc[80-59:81].mean(axis=1)
print(P.shape,R.notna().sum().to_dict(),m.notna().sum(),R.iloc[21:81].notna().sum().to_dict())
for s in P:
 x=R[s].iloc[21:81];z=m;ok=x.notna()&z.notna();print(s,ok.sum())
