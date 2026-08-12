import sys; sys.path.insert(0,'scripts')
import inspect
import factor_research_lib as frl
print(inspect.getsource(frl.full_eval))
print("===LOAD_PANELS===")
print(inspect.getsource(frl.load_panels)[:1500])