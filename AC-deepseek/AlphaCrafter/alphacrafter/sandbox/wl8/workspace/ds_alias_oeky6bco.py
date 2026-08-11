import inspect
from alphacrafter.sim.utils.rebalance_to_weights import evaluate_trade
import alphacrafter.sim.utils.rebalance_to_weights as m
src = inspect.getsource(m)
# print the whole module source (it's small enough probably)
print(src[:6000])
