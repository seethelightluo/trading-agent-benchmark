import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices

prices = load_prices(days=2500)
s = 'SPX'
df = prices[s]
print("index name:", df.index.name)
c = df['close']
y = df.index.year.values
m = df.index.month.values
g = df.groupby([df.index.year, df.index.month])['close'].last()
print("groupby head:")
print(g.head())
print("index:", g.index, "names:", g.index.names)
