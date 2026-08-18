import sys, importlib.util
spec = importlib.util.spec_from_file_location("strat", "strategy.py")
strat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strat)

s = strat._series(strat._stock("SPX"))
print("SPX series len:", None if s is None else len(s), "tail:", None if s is None else s.tail(3).tolist())
print("SPX last is nan:", s is not None and s.iloc[-1] != s.iloc[-1])

s2 = strat._series(strat._stock("XAU"))
print("XAU series tail:", None if s2 is None else s2.tail(3).tolist())

# replicate factor score for all assets to see differentiation
acc = __import__("alphacrafter.sim.utils", fromlist=["get_account_dict"]).get_account_dict()
assets = acc.get("watch_list", [])
series = {a: strat._series(strat._stock(a)) for a in assets}
usable = {a: s.pct_change().rename(a) for a, s in series.items() if s is not None and s.iloc[-1] == s.iloc[-1]}
print("usable count:", len(usable))
import pandas as pd
R = pd.concat(usable, axis=1, join="inner").dropna().tail(150)
print("R shape:", R.shape, "R last row any nan:", R.iloc[-1].isna().any())
cp = (1.0 + R).cumprod()
mom = cp.shift(5) / cp.shift(25) - 1.0
rel_mom = mom.sub(mom.median(axis=1), axis=0)
print("rel_mom last row:", rel_mom.iloc[-1].round(4).to_dict())
