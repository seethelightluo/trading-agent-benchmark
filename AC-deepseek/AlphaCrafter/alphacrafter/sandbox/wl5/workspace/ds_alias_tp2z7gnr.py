import sys, json, os
sys.path.insert(0,'scripts')
import miner3_lib as L
C, V, H, Lw, O = L.load_close_panel(4000)
print("Panel:", C.index.min().date(), "->", C.index.max().date(), "|", len(C), "dates x", C.shape[1], "assets")
print("Assets:", list(C.columns))
# check recent data availability
print("\nLast 3 rows:")
print(C.tail(3).to_string())
print("\nMissing close frac:", C.isna().mean().round(3).to_dict())
