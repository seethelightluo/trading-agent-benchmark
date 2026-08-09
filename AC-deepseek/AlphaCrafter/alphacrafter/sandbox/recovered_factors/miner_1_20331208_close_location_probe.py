import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
acct=get_account_dict(); print(acct['watch_list'])
for s in acct['watch_list'][:2]:
 d=get_stock_daily_data(s, 3000)
 print(s, type(d), getattr(d,'shape',None), list(d.columns), d.tail(3).to_string())
