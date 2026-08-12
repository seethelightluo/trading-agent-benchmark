from pathlib import Path
import numpy as np
shapes = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    arr = np.load(p, allow_pickle=False)
    shapes.setdefault(arr.shape[0], []).append(p.name.replace('_signal.npy',''))
for k, v in sorted(shapes.items()):
    print(k, len(v), v)
# Check if any factor json references these artifacts
import json
for p in Path('factors').glob('*.json'):
    try:
        d = json.load(open(p))
        if 'signal_artifact' in d:
            pass
    except Exception:
        pass
print("total npy:", len(list(Path('factors').glob('*_signal.npy'))))
print("total json:", len(list(Path('factors').glob('*.json'))))