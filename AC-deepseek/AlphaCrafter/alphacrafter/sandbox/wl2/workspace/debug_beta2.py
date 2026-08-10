import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner_1_lib import load_panel, macro_series, TRADABLES
panel = load_panel()
vix = macro_series('VIX').pct_change()
beta_parts = {}
for a in panel.columns:
    s = panel[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename('a'), vix.reindex(ar.index).rename('v')], axis=1).dropna()
    b = df['a'].rolling(60).cov(df['v']) / df['v'].rolling(60).var()
    r = b.reindex(panel.index)
    beta_parts[a] = r
    print(a, 'b len', len(b), 'r len', len(r), 'r name', r.name)
bp = pd.DataFrame(beta_parts, index=panel.index)
print('beta_panel', bp.shape)
print('columns sample:', bp.columns.tolist()[:20])
print('nunique cols:', bp.columns.nunique())
