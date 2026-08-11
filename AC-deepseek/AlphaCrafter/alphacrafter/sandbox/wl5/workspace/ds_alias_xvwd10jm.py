
import sys
sys.path.insert(0, 'scripts')
import miner3_lib as L
import pandas as pd

C, V, H, Lo, O = L.load_close_panel(4000)
print("Panel rows:", len(C), "cols:", C.shape[1])
print("Date range:", C.index.min().date(), "->", C.index.max().date())
print("Last 5 dates:")
print(C.tail(5).index.tolist())
print("\nVolume data available:")
print(V.tail(3).to_string())
print("\nNaN check last row:", C.iloc[-1].isna().sum(), "of", C.shape[1])
