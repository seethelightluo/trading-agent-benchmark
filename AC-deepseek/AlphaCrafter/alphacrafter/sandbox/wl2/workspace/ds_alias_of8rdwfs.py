import sys
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import ASSETS, load_asset, load_macro
import pandas as pd
pd.set_option('display.width', 250)
for s in ASSETS:
    df = load_asset(s, days=2600)
    if df is None: 
        print(s, 'NO DATA'); continue
    c = df['close']
    r20 = c.iloc[-1]/c.iloc[-21]-1
    r60 = c.iloc[-1]/c.iloc[-61]-1
    r120 = c.iloc[-1]/c.iloc[-121]-1
    print(f"{s:<12} last={c.iloc[-1]:>12.2f} r20={r20:>8.1%} r60={r60:>8.1%} r120={r120:>8.1%} rows={len(df)}")
print('--- macros ---')
for m in ['VIX','DXY','USDJPY','EURUSD','USDCNY']:
    x = load_macro(m)
    if x is None: continue
    print(f"{m:<8} last={x.iloc[-1]:>10.2f} r20={x.iloc[-1]/x.iloc[-21]-1:>8.1%} r60={x.iloc[-1]/x.iloc[-61]-1:>8.1%}")