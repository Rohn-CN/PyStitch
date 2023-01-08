import numpy as np


class A:
    def __init__(self, b):
        self.b = b
    def func(self,x):
        y = x
        y[0] = self.b


if __name__ == '__main__':
    x = np.array([1,2,3,1])
    for i in range(20):
        a = A(i)
        a.func(x)
    for i in range(len(x)):
        print(x[i])
