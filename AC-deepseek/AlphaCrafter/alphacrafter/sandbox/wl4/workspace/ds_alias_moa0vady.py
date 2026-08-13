import sys; sys.path.insert(0, 'scripts')
import inspect, factor_lib
print(inspect.getsource(factor_lib)[:4000])