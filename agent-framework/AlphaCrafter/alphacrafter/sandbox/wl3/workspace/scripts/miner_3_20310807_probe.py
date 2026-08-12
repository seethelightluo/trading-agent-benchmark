from alphacrafter.sim.utils import get_index_daily_data
for n in [100,1000,5000]:
 x=get_index_daily_data('SPX',days=n); print(n, None if x is None else len(x))
