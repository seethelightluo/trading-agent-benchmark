"""Inspect data coverage per asset."""
import sys
sys.path.insert(0, 'scripts')
from miner3_20270128_common import load_data, build_panel
uni = load_data()
for s,df in uni.items():
    print(s, len(df), str(df.index.min().date()), str(df.index.max().date()))
close, ret = build_panel(uni)
print("close shape", close.shape)
print("close last date", close.index.max().date())
print("cols", list(close.columns))
# how many rows notna per col
print(close.notna().sum())