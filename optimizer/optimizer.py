#written by Michael chen
import numpy as np


class arguments(object):
    def __init__(self,gamma=0.9,eta=10e-4,eps=10e-8):
        self.gamma = gamma
        self.eta = eta
        self.eps = eps
class optFunction(object):
    def __init__(self,name,arguments):
        self.name = name
        self.arguments=arguments
    #updata net's arguments according to grad
    def updateData(self,data,grad,g):
        if self.name=='momentum':
            V=0
            dWprev=np.zeros(grad.size)
            for i in range(grad.size):
                V = self.arguments.gamma*dWprev[i]+self.arguments.eta*(grad[i]+data[i]*self.arguments.eta)
                data[i]=data[i]-V
                dWprev[i]=V
        elif self.name=='adagrad':
            for i in range(grad.size):
                g[i]=g[i]+grad[i]*grad[i]
                data[i]=data[i]-self.arguments.eta*grad[i]/(np.sqrt(g[i])+self.arguments.eps)
        elif self.name=='adadelta':
            for i in range(grad.size):
                g[i]=self.arguments.gamma*g[i]+(1-self.arguments.gamma)*grad[i]*grad[i]
                data[i]=data[i]-self.arguments.eta*grad[i]/(np.sqrt(g[i])+self.arguments.eps)
        elif self.name=='gradientdescent':
            for i in range(grad.size):
                data[i]=data[i]-self.arguments.eta*grad[i]
        return data,g
