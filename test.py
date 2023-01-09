import numpy as np


class A:
    def __init__(self):
        pass
    def func(self,x):
        x = np.array([10])
    def func2(self,x):
        x[0] = 100
    def set_b(self,b):
        self.b = b
class B:
    def __init__(self):
        pass
    def fun(self,x):
        y = np.array([100,20])
        x[:] = y[:]


if __name__ == '__main__':
    x = np.array([1,2,3,1])
    y = np.array([2,3])
    a = A()
    a.set_b(x)
    b = B()
    b.fun(a.b)
    print(a.b)