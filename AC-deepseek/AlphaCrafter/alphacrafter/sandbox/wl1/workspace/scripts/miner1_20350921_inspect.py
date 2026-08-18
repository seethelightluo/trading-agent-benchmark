import pickle, os, sys
for f in ['panel_cache_20350907.pkl','panel_cache_20350906.pkl','panel_cache_20350824.pkl','panel_cache_20350823.pkl','panel_cache_20350726.pkl']:
    if os.path.exists(f):
        with open(f,'rb') as fh:
            d = pickle.load(fh)
        print(f, type(d), 'size', os.path.getsize(f))
        if isinstance(d, dict):
            print('  keys:', list(d.keys())[:30])
            for k in list(d.keys())[:3]:
                v = d[k]
                print('   ', k, type(v), getattr(v, 'shape', None))
        elif hasattr(d, 'shape'):
            print('  shape', d.shape, 'columns', list(d.columns)[:20] if hasattr(d,'columns') else '')
        print()
