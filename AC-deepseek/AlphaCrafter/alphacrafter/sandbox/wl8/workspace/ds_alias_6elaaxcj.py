import inspect, rebalance_to_weights as rw
print(rw.__file__)
src = inspect.getsource(rw)
print(src[:2000])
