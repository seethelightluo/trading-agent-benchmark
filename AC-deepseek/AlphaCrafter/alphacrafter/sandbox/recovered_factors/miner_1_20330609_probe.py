import os, json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
acct=get_account_dict(); print(acct['watch_list'])
for fn,name in [(get_stock_daily_data,'stock'),(get_index_daily_data,'index')]:
    try:
        d=fn('SPX')
        print(name, type(d), getattr(d,'shape',None), getattr(d,'columns',None)); print(d.tail(3))
    except Exception as e: print(name,repr(e))
print('factor files',len(os.listdir('factors')))
for f in os.listdir('factors')[:3]: print(f)
