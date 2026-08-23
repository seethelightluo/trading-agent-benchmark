import sys, json, inspect
sys.path.insert(0,'scripts')
import miner_shared as m
# print sources of key functions
for fn in ['forward_ret','daily_ic','ic_stats','rank_turnover','coverage_stats','library_panel','load_close','load_macro']:
    try:
        src = inspect.getsource(getattr(m, fn))
        print('='*20, fn, '='*20)
        print(src[:2000])
    except Exception as e:
        print(fn, 'ERR', e)