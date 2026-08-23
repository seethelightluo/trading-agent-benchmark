"""miner_3 2034-05-11 probe: confirm data availability range."""
import sys, json
sys.path.insert(0, 'scripts')
import factor_validation_lib as fvl

p = fvl.load_panel(max_date='2034-05-10')
print("tail date", p.index.max(), "rows", len(p), "cols", p.shape[1])
print("first date", p.index.min())
print("n per last date", p.notna().iloc[-1].sum())
print("assets", list(p.columns))

m = fvl.load_macro('VIX', '2034-05-10')
print("VIX tail", m.index.max(), list(m.tail(3)))