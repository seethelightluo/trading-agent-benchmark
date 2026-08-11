import inspect
from alphacrafter.sim.utils.rebalance_to_weights import evaluate_trade, one_way_turnover
print(inspect.getsource(evaluate_trade))
print('=====')
print(inspect.getsource(one_way_turnover))