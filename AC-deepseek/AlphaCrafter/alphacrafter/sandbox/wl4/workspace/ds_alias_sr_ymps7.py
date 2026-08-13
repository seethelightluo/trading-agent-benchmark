lines = open('memory.txt').readlines()
print('total lines:', len(lines))
for i, l in enumerate(lines):
    if l.startswith('2032'):
        print(i, l[:100])
