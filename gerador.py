# gerador de matriz 
import numpy as np

size = 9
low = 22
high = 37
step = 2

matrix = np.random.choice([x for x in xrange(low,high,step)],size*size)
matrix.resize(size,size)

matrix2 = np.random.choice([x for x in xrange(low,high,step)],size*size)
matrix2.resize(size,size)

print size 
for i in range(size):
    print ' '.join(map(str, matrix[i]))
    
for i in range(size):
    print ' '.join(map(str, matrix2[i]))