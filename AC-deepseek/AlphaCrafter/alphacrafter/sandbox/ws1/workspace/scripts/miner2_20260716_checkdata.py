import sys
sys.path.insert(0, 'scripts')
from miner2_20260716_common import load_panel, load_index_panel, MIN_INSTR, CUTOFF
panel = load_panel()
idx = load_index_panel()
print('CUTOFF', CUTOFF)
print('panel shape', panel.shape)
print('idx shape', idx.shape)
print('panel cols', panel.columns.tolist())
print('idx cols', idx.columns.tolist())
print('panel index min/max', panel.index.min(), panel.index.max())
print('idx index min/max', idx.index.min(), idx.index.max())
print(panel.tail(3))
