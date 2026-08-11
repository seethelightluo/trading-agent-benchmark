import inspect, sys
sys.path.insert(0, 'scripts')
import factor_validate as fv
src = inspect.getsource(fv)
print(src[:6000])