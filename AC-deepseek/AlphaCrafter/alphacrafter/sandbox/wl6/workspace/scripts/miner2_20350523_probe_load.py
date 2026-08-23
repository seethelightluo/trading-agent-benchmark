import sys, time
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel
t0 = time.time()
px = load_panel(max_date="2035-05-22")
print("load_panel", px.shape, "elapsed", round(time.time()-t0, 2))