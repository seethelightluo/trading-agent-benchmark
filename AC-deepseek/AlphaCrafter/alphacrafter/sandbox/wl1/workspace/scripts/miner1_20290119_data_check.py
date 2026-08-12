"""miner_1 sanity check: data availability through current date."""
import sys
sys.path.insert(0, "scripts")
from miner1_common import load_closes, load_macro

px = load_closes(end_date="2029-01-18")
mx = load_macro(end_date="2029-01-18")
print("price panel:", px.shape, px.index.min().date(), "->", px.index.max().date())
print("macro panel:", mx.shape, mx.index.min().date(), "->", mx.index.max().date())
print("\nlast close per asset (as of", px.index.max().date(), "):")
print(px.tail(1).T)
print("\nn_dates:", len(px))
print("coverage per asset:")
print(px.notna().mean())
