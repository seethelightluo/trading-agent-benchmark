import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']
def x(s,idx=False):
 d=(get_index_daily_data(s,5000) if idx else get_stock_daily_data(s,5000)).copy();d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame({a:x(a) for a in A});r=P.pct_change(); d=x('DXY',True).reindex(P.index).pct_change()
print('P',P.index.min(),P.index.max(),len(P),'D',d.notna().sum(),(d<0).sum(),(d>0).sum(),d.dtype)
z=r[A[0]].where(d<0);print('MASK',z.notna().sum(),z.dropna().head())
a=r[A[0]].where(d<0).rolling(60,min_periods=35).mean();b=r[A[0]].where(d>0).rolling(60,min_periods=35).mean();print('ROLL',a.notna().sum(),b.notna().sum(),(a-b).notna().sum())
