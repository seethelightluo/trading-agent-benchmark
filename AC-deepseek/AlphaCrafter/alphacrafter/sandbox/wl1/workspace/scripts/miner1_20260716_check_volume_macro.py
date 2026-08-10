"""miner_1: quick check of volume data quality and macro panel alignment."""
import pandas as pd, numpy as np

panel = pd.read_pickle('scripts/panel_cache.pkl')
close, vol, macro = panel['close'], panel['vol'], panel['macro']

print("=== volume quality (fraction of rows with valid & nonzero) ===")
v = vol.replace(0, np.nan)
print((v.notna().mean() * 100).round(1).to_string())

print()
print("=== macro columns ===", list(macro.columns))
print("macro range:", macro.index.min(), "->", macro.index.max(), "rows:", len(macro))
print(macro.notna().mean().round(3).to_string())

print()
wi = close.index[close.index.dayofweek >= 5]
wd = close.index[close.index.dayofweek < 5]
print("weekend rows:", len(wi), "| with >=8 valid:", int(close.loc[wi].notna().sum(axis=1).ge(8).sum()))
print("weekday rows:", len(wd), "| with >=8 valid:", int(close.loc[wd].notna().sum(axis=1).ge(8).sum()))

# per-asset weekday vs total valid
print()
print("per-asset valid counts (weekday / total):")
for c in close.columns:
    print(f"  {c:10s} {int(close.loc[wd,c].notna().sum()):5d} / {int(close[c].notna().sum()):5d}")
