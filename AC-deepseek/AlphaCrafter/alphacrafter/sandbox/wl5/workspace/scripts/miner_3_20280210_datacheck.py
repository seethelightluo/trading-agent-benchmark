# -*- coding: utf-8 -*-
"""miner_3 2028-02-10 datacheck: verify visible data window and quirks."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
print('Panel dates: %s -> %s | n_dates=%d n_assets=%d' % (
    C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))
print('Last 5 dates:')
print(C.index[-5:].strftime('%Y-%m-%d').tolist())

R = C.pct_change()
print('\nMissing ratios (close):')
print((C.isna().mean().round(4)).to_dict())
print('\nMissing ratios (open):')
print((O.isna().mean().round(4)).to_dict())
print('\nVolume zero/na ratios:')
print(((V.fillna(0) == 0).mean().round(3)).to_dict())
print('\nHigh/Low missing ratios:')
print(((H.isna() | Lw.isna()).mean().round(3)).to_dict())

# latest 10d forward return sample for sanity
print('\nLatest close row (%s):' % C.index.max().date())
print(C.iloc[-1].round(4).to_dict())
print('\nLatest 10d returns (%s):' % C.index.max().date())
print(R.iloc[-1].round(4).to_dict())
