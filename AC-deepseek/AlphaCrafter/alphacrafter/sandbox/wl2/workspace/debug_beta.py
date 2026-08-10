import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner_1_lib import load_panel, macro_series, TRADABLES
panel = load_panel()
vix = macro_series('VIX').pct_change()
for a in TRADABLES:
    try:
        s = panel[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename('a'), vix.reindex(ar.index).rename('v')], axis=1).dropna()
        b = df['a'].rolling(60).cov(df['v']) / df['v'].rolling(60).var()
        if not isinstance(b, pd.Series) or b.shape[0] != len(df):
            print('PROBLEM', a, type(b), getattr(b, 'shape', None))
            if isinstance(b, pd.DataFrame):
                print('  cols:', b.columns.tolist()[:8], b.shape)
    except Exception as e:
        print('ERR', a, repr(e))
print('done')
