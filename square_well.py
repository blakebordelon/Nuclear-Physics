import numpy as np


def func_j(x):
	return np.sin(x) / x

def dfunc_j(x):
	return np.cos(x)/x - np.sin(x) / (x*x)

T = 50
x = 1

for t in range(T):

	x -= func_j(x) / dfunc_j(x)	
	print(x)


print("final x")
print(x)