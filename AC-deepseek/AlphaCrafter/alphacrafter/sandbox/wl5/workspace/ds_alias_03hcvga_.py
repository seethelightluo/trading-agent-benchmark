import os
print("CWD:", os.getcwd())
for p in ['.', '../', '../../']:
    try:
        print(p, os.listdir(p)[:20])
    except Exception as e:
        print(p, "ERR", e)