import sys
print(sys.path)
try:
    from alphacrafter.sim.utils import get_stock_daily_data
    print("import ok")
except Exception as e:
    print("ERR", e)