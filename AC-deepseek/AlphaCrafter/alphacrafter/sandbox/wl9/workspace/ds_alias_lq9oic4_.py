import pandas as pd, os
from pathlib import Path
SD = Path('../persistent/stock_data')
ID = Path('../persistent/index_data')
df = pd.read_csv(SD/'SPX.csv')
print("SPX cols:", list(df.columns), "rows:", len(df))
print(df.head(3).to_string())
print(df.tail(2).to_string())
ini = pd.read_csv(ID/'VIX.csv')
print("VIX cols:", list(ini.columns), "rows:", len(ini))
print(ini.tail(2).to_string())
for a in ['BTC','XAU','CN10Y']:
    d = pd.read_csv(SD/f'{a}.csv')
    print(a, list(d.columns), len(d))
# check volume availability
print("volume non-na SPX:", df['volume'].notna().sum() if 'volume' in df.columns else 'no vol col')