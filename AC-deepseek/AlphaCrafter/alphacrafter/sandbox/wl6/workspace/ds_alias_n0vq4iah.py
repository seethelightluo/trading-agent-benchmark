import os
# Check the index data formats
for f in ['VIX.csv','DXY.csv']:
    p = os.path.join('../persistent/index_data', f)
    with open(p) as fh:
        lines = fh.readlines()
    print(f, 'lines:', len(lines))
    print('head:', lines[:3])
    print('tail:', lines[-2:])
    print('---')