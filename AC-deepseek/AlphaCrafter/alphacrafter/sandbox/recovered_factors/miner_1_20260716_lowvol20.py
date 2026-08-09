# [line 1 missing]
# [line 2 missing]
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; pcols={}; fac={}; rav={}; rev={}
# [line 5 missing]
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)