import numpy as np
from activation.activation import acti_func


class FcLayer(object):
    def __init__(self,inputsize,outputsize,acti,name='fc'):
        self.inputsize = inputsize
        self.outputsize=outputsize
        self.W=np.zeros([outputsize,inputsize])
        self.dW=np.zeros([outputsize,inputsize])
        self.b=np.zeros([1,outputsize])
        self.db=np.zeros([1,outputsize])
        self.inputdeltas=np.zeros([1,inputsize])
        self.outputdeltas=np.zeros(outputsize)
        self.inputdata=np.zeros([1,inputsize])
        self.outdata=np.zeros([1,outputsize])
        self.acti=acti
        self.name = name

        base=np.sqrt(6.0/(self.outputsize+self.inputsize))
        self.W=(np.random.random(self.W.size)*2*base-base).reshape(self.outputsize,self.inputsize)
        self.b=(np.random.random(self.outputsize)*2*base-base).reshape(1,self.outputsize)
    def forwardpropagation(self,data_x):
        self.inputdata=data_x
        datasize=data_x.shape[0]
        bb=self.b
        bbb=np.repeat(bb,datasize,0)
        self.outputdata=self.acti.f(np.dot(data_x,self.W.T)+bbb)
        return self.outputdata
    def backwardpropagation(self,data_x,delta_last,prev_acti):
        datasize=data_x.shape[0]
        self.outputdeltas=delta_last
        self.inputdeltas=np.dot(self.outputdeltas,self.W)*prev_acti.f_(data_x)
        self.dW=np.array(np.dot(np.matrix(self.outputdeltas).T,np.matrix(data_x))/datasize)
        self.db=np.mean(self.outputdeltas,0)
        return self.inputdeltas
