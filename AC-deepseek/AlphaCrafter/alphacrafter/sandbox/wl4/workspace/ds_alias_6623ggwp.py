import json
print(open('factor_ensemble.json').read())
print('---memory head---')
lines = open('memory.txt').read().splitlines()
print('total lines:', len(lines))
print('first 3 lines:')
for l in lines[:3]:
    print(l[:300])
print('---lines with 2034---')
for l in lines:
    if '2034' in l:
        print(l[:250])
        print('~~~')
