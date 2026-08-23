import sys; sys.path.insert(0, 'scripts')
from miner2_20291009_lib import load_prices
prices, closes = load_prices(4000)
print("closes shape:", closes.shape)
print("first date:", closes.index[0], "last date:", closes.index[-1])
print("cols:", list(closes.columns))