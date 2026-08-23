import json, os
print(os.listdir('factors'))
print('--- non-bak files ---')
for f in os.listdir('factors'):
    if not f.endswith('.bak') and os.path.isfile(os.path.join('factors', f)):
        print(f, os.path.getsize(os.path.join('factors', f)))