import numpy as np

class acti_func(object):
    def __init__(self,type):
        self.type=type
    def f(self,data):
        if self.type=='tanh':
            r=2.0/(1.0+np.exp(-2.0*data))-1.0
        elif self.type=='Relu':
            r=np.maximum(data,0.0)
        elif self.type=='identity':
            r=data
        elif self.type=='sigmoid':
            r=1.0/(1.0+np.exp(-data))
        return r
    def f_(self,y):
        if self.type=='tanh':
            r=1.0-y*y
        elif self.type=='Relu':
            r=np.where(y>=0.0,1.0,0.0)
        elif self.type=='identity':
            r=np.ones(y.shape)
        elif self.type=='sigmoid': 
            r=y*(1.0-y)
        return r
