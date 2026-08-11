# Trader: evaluate_trade + Decision internals (gate math)
import sys, inspect
sys.path.insert(0, ".")
from alphacrafter.sim.utils import rebalance_to_weights, evaluate_trade
import alphacrafter.sim.utils as U

print("=== rebalance_to_weights lines 70-160 ===")
src = inspect.getsource(rebalance_to_weights)
for i, ln in enumerate(src.split("\n")[70:165], start=70):
    print(f"{i:3d}| {ln}")

print("\n=== evaluate_trade source ===")
print(inspect.getsource(evaluate_trade))

# also check Decision class
for name in dir(U):
    if "ecision" in name or "rade" in name:
        obj = getattr(U, name)
        if callable(obj) or isinstance(obj, type):
            try:
                s = inspect.getsource(obj)
                if len(s) < 6000:
                    print(f"\n=== {name} ===")
                    print(s[:5000])
            except Exception:
                pass
