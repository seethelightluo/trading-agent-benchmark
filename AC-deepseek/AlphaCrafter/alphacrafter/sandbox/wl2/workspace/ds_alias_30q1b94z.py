import sys
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import load_asset
for s in ['SX5E','BTC','US10Y','CN10Y','ETH']:
    df = load_asset(s, days=2600)
    c = df['close']
    tail = c.iloc[-15:]
    print(s, 'tail:', [f'{v:.2f}' for v in tail.values])
    print('   last 20 unique:', c.iloc[-20:].nunique(), 'nan_last10:', c.iloc[-10:].isna().sum())