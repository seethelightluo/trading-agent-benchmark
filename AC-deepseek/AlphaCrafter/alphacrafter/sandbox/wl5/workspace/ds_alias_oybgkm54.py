import inspect, sys
sys.path.insert(0, 'scripts')
import miner3_lib
src = inspect.getsource(miner3_lib)
print(src[:5000])