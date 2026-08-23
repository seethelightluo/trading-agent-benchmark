import os
print("MEM len:", os.path.getsize('memory.txt'))
mems=open('memory.txt').read()
print(repr(mems[-2500:]))
print("===AUDIT tail===")
try:
    lines=open('factor_library_audit.jsonl').read().strip().split('\n')
    for l in lines[-5:]:
        print(l[:1500])
except Exception as e:
    print("audit err",e)